"""Immutable P1 project/topology contracts.

These contracts deliberately model project metadata and stable provenance only.
They do not read or write project files, construct viewer caches, execute a
scientific stage, or load a prediction provider.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any
from uuid import UUID

from core.contracts._json import json_safe_mapping
from core.contracts.frv import (
    SourceIdentity,
    frv_digest,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)


PROJECT_SCHEMA_VERSION = "treeline-project-v1"
PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION = "treeline-project-source-reference-v1"
COORDINATE_CONTEXT_SCHEMA_VERSION = "treeline-coordinate-context-v1"
TOWER_SCHEMA_VERSION = "treeline-tower-v1"
TOWER_REFINEMENT_SCHEMA_VERSION = "treeline-tower-refinement-v1"
SPAN_SCHEMA_VERSION = "treeline-span-v1"
SPAN_GEOMETRY_SCHEMA_VERSION = "treeline-span-geometry-v1"
SPAN_ANALYSIS_SCHEMA_VERSION = "treeline-span-analysis-v1"
VIEWER_REFERENCE_SCHEMA_VERSION = "treeline-viewer-reference-v1"
PREDICTION_CAPABILITY_SCHEMA_VERSION = "treeline-prediction-capability-v1"
PREDICTION_PROVIDER_SCHEMA_VERSION = "treeline-prediction-provider-v1"
PREDICTION_REQUEST_SCHEMA_VERSION = "treeline-prediction-request-v1"
PREDICTION_RESULT_SCHEMA_VERSION = "treeline-prediction-result-v1"
PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION = "treeline-prediction-artifact-reference-v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOWER_ID_RE = re.compile(r"T([0-9]{3,})\Z")
_SPAN_ID_RE = re.compile(r"S([0-9]{3,})\Z")
_RUN_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_CACHE_ID_RE = re.compile(r"[a-z][a-z0-9._-]{0,127}\Z")
_PROVIDER_ID_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*\Z")
_WINDOWS_RESERVED_DEVICE_BASENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_WINDOWS_FORBIDDEN_FILENAME_CHARACTERS = frozenset('<>"|?*')
_MAX_METADATA_NODES = 512
_MAX_METADATA_SEQUENCE_LENGTH = 128


class CoordinateFrame(str, Enum):
    """Coordinate frames accepted by P1 project records."""

    SOURCE_LAS_ABSOLUTE = "SOURCE_LAS_ABSOLUTE"


class CoordinateUnit(str, Enum):
    METRE = "m"


class TowerLifecycleState(str, Enum):
    CONFIRMED = "CONFIRMED"
    ARCHIVED = "ARCHIVED"


class TowerRefinementStatus(str, Enum):
    DEFENSIBLE = "DEFENSIBLE"


class TowerRefinementQuality(str, Enum):
    MANUAL_CONFIRMED = "MANUAL_CONFIRMED"


class SpanTopologyState(str, Enum):
    CONFIRMED = "CONFIRMED"
    ARCHIVED = "ARCHIVED"


class AnalysisStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    STALE = "STALE"


class AnalysisStage(str, Enum):
    GROUND = "ground"
    WIRE = "wire"
    VEGETATION = "vegetation"
    TREES = "trees"
    CLEARANCE = "clearance"
    RISK = "risk"
    REPORT = "report"


class ViewerCacheState(str, Enum):
    NOT_BUILT = "NOT_BUILT"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    STALE = "STALE"


class PredictionCapabilityState(str, Enum):
    NOT_CONNECTED = "NOT_CONNECTED"
    AVAILABLE = "AVAILABLE"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INCOMPATIBLE = "INCOMPATIBLE"


class PredictionExecutionMode(str, Enum):
    FILE_EXCHANGE = "file_exchange"
    LOCAL_API = "local_api"
    SUBPROCESS = "subprocess"
    PLUGIN = "plugin"
    REMOTE = "remote"


def _exact_string(value: Any, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(
            f"{name} must be a non-empty built-in string without surrounding whitespace"
        )
    return value


def _sha256(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return text


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive built-in integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative built-in integer")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite numeric value")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a numeric sequence")
    if len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return tuple(_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value))


def _xyz(value: Any, name: str) -> tuple[float, float, float]:
    result = _vector(value, 3, name)
    return result[0], result[1], result[2]


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _timestamp(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    return text


def _ordered_unique_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a sequence of strings")
    items = tuple(_exact_string(item, f"{name}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates")
    return items


def _ordered_unique_enums(value: Any, enum_type: type[Enum], name: str) -> tuple[Enum, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a sequence")
    items = tuple(_enum(item, enum_type, f"{name}[]") for item in value)
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates")
    return items


def _freeze_metadata(value: Any, name: str, node_count: list[int]) -> Any:
    node_count[0] += 1
    if node_count[0] > _MAX_METADATA_NODES:
        raise ValueError(f"{name} exceeds the bounded metadata size")
    if isinstance(value, Mapping):
        if len(value) > _MAX_METADATA_SEQUENCE_LENGTH:
            raise ValueError(f"{name} has too many mapping entries")
        return MappingProxyType(
            {
                key: _freeze_metadata(value[key], f"{name}.{key}", node_count)
                for key in sorted(value)
            }
        )
    if isinstance(value, list | tuple):
        if len(value) > _MAX_METADATA_SEQUENCE_LENGTH:
            raise ValueError(f"{name} has too many sequence entries")
        return tuple(
            _freeze_metadata(item, f"{name}[]", node_count) for item in value
        )
    return value


def _immutable_metadata(value: Mapping[str, Any] | None, name: str) -> Mapping[str, Any]:
    safe = json_safe_mapping(value, name)
    return _freeze_metadata(safe, name, [0])


def _thaw_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_metadata(item) for item in value]
    return value


def _safe_relative_posix_path(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if any(ord(character) < 32 or ord(character) == 127 for character in text):
        raise ValueError(f"{name} must not contain ASCII control characters")
    if "\\" in text or ":" in text:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or str(path) != text
        or any(part.endswith((".", " ")) for part in path.parts)
    ):
        raise ValueError(f"{name} must be a traversal-safe normalized relative POSIX path")
    for component in path.parts:
        if any(character in _WINDOWS_FORBIDDEN_FILENAME_CHARACTERS for character in component):
            raise ValueError(f"{name} must not contain Windows-forbidden filename characters")
        windows_basename = component.split(".", maxsplit=1)[0].rstrip(". ").upper()
        if windows_basename in _WINDOWS_RESERVED_DEVICE_BASENAMES:
            raise ValueError(f"{name} must not contain a Windows reserved device basename")
    return text


def _display_id(value: Any, pattern: re.Pattern[str], prefix: str, name: str) -> str:
    text = _exact_string(value, name)
    matched = pattern.fullmatch(text)
    if matched is None or int(matched.group(1)) <= 0:
        raise ValueError(f"{name} must be a canonical {prefix} display ID such as {prefix}001")
    return text


def tower_display_id_for_sequence(sequence: int) -> str:
    """Return the immutable project-local display ID for a persisted sequence value."""

    return f"T{_positive_int(sequence, 'sequence'):03d}"


def span_display_id_for_sequence(sequence: int) -> str:
    """Return the immutable project-local display ID for a persisted sequence value."""

    return f"S{_positive_int(sequence, 'sequence'):03d}"


def _run_id(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if _RUN_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lower-case opaque run identifier")
    return text


def _opaque_id(value: Any, name: str) -> str:
    return _run_id(value, name)


@dataclass(frozen=True, slots=True)
class CoordinateContext:
    """Validated source-absolute coordinate declaration bound to a source identity."""

    schema_version: str
    frame: CoordinateFrame
    units: CoordinateUnit
    origin_xyz: tuple[float, float, float]
    source_content_identity_digest: str
    source_identity_record_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != COORDINATE_CONTEXT_SCHEMA_VERSION:
            raise ValueError("unsupported coordinate-context schema version")
        _enum(self.frame, CoordinateFrame, "frame")
        _enum(self.units, CoordinateUnit, "units")
        object.__setattr__(self, "origin_xyz", _xyz(self.origin_xyz, "origin_xyz"))
        _sha256(self.source_content_identity_digest, "source_content_identity_digest")
        _sha256(self.source_identity_record_digest, "source_identity_record_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame": self.frame.value,
            "units": self.units.value,
            "origin_xyz": list(self.origin_xyz),
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
        }


@dataclass(frozen=True, slots=True)
class SourceIdentityBinding:
    """Digest/context reference to the repository's authoritative ``SourceIdentity``."""

    source_content_identity_digest: str
    source_identity_record_digest: str
    coordinate_context: CoordinateContext

    def __post_init__(self) -> None:
        _sha256(self.source_content_identity_digest, "source_content_identity_digest")
        _sha256(self.source_identity_record_digest, "source_identity_record_digest")
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        if (
            self.coordinate_context.source_content_identity_digest
            != self.source_content_identity_digest
            or self.coordinate_context.source_identity_record_digest
            != self.source_identity_record_digest
        ):
            raise ValueError("coordinate_context must bind to the same source identity")

    @classmethod
    def from_source_identity(cls, source_identity: SourceIdentity) -> SourceIdentityBinding:
        if type(source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        validate_source_content_identity_digest(source_identity)
        validate_source_identity_record_digest(source_identity)
        return cls(
            source_content_identity_digest=source_identity.source_content_identity_digest,
            source_identity_record_digest=source_identity.source_identity_record_digest,
            coordinate_context=CoordinateContext(
                schema_version=COORDINATE_CONTEXT_SCHEMA_VERSION,
                frame=CoordinateFrame(source_identity.coordinate_frame),
                units=CoordinateUnit(source_identity.units),
                origin_xyz=source_identity.origin_context,
                source_content_identity_digest=source_identity.source_content_identity_digest,
                source_identity_record_digest=source_identity.source_identity_record_digest,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "coordinate_context": self.coordinate_context.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProjectSourceReference:
    """Project source reference that reuses the existing complete ``SourceIdentity``."""

    schema_version: str
    source_identity: SourceIdentity
    coordinate_context: CoordinateContext

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION:
            raise ValueError("unsupported project source-reference schema version")
        if type(self.source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        validate_source_content_identity_digest(self.source_identity)
        validate_source_identity_record_digest(self.source_identity)
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        binding = SourceIdentityBinding.from_source_identity(self.source_identity)
        if self.coordinate_context != binding.coordinate_context:
            raise ValueError("coordinate_context must exactly match SourceIdentity")

    @property
    def binding(self) -> SourceIdentityBinding:
        return SourceIdentityBinding.from_source_identity(self.source_identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_identity": self.source_identity.to_dict(),
            "coordinate_context": self.coordinate_context.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ViewerReference:
    """Abstract cache state/reference; it intentionally contains no physical path."""

    schema_version: str
    state: ViewerCacheState
    active_cache_id: str | None

    def __post_init__(self) -> None:
        if self.schema_version != VIEWER_REFERENCE_SCHEMA_VERSION:
            raise ValueError("unsupported viewer-reference schema version")
        _enum(self.state, ViewerCacheState, "state")
        if self.active_cache_id is None:
            if self.state in {ViewerCacheState.AVAILABLE, ViewerCacheState.STALE}:
                raise ValueError("available or stale viewer state requires active_cache_id")
        else:
            cache_id = _exact_string(self.active_cache_id, "active_cache_id")
            if _CACHE_ID_RE.fullmatch(cache_id) is None:
                raise ValueError("active_cache_id must be a lower-case opaque cache identifier")
            if self.state in {ViewerCacheState.NOT_BUILT, ViewerCacheState.UNAVAILABLE}:
                raise ValueError("viewer state does not permit active_cache_id")

    @classmethod
    def not_built(cls) -> ViewerReference:
        return cls(
            schema_version=VIEWER_REFERENCE_SCHEMA_VERSION,
            state=ViewerCacheState.NOT_BUILT,
            active_cache_id=None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "active_cache_id": self.active_cache_id,
        }


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Digest-checked project-relative reference, without any filesystem access."""

    relative_path: str
    role: str
    sha256: str
    byte_size: int
    schema_version: str

    def __post_init__(self) -> None:
        _safe_relative_posix_path(self.relative_path, "relative_path")
        _exact_string(self.role, "role")
        _sha256(self.sha256, "sha256")
        _nonnegative_int(self.byte_size, "byte_size")
        _exact_string(self.schema_version, "schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class TowerGeometry:
    base_xyz_absolute_m: tuple[float, float, float]
    top_xyz_absolute_m: tuple[float, float, float]
    height_m: float

    def __post_init__(self) -> None:
        base = _xyz(self.base_xyz_absolute_m, "base_xyz_absolute_m")
        top = _xyz(self.top_xyz_absolute_m, "top_xyz_absolute_m")
        height = _finite_number(self.height_m, "height_m")
        if height <= 0.0:
            raise ValueError("height_m must be positive")
        vertical_height = top[2] - base[2]
        if vertical_height <= 0.0 or not math.isclose(
            height, vertical_height, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError("height_m must equal the positive base-to-top Z difference")
        object.__setattr__(self, "base_xyz_absolute_m", base)
        object.__setattr__(self, "top_xyz_absolute_m", top)
        object.__setattr__(self, "height_m", height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_xyz_absolute_m": list(self.base_xyz_absolute_m),
            "top_xyz_absolute_m": list(self.top_xyz_absolute_m),
            "height_m": self.height_m,
        }


@dataclass(frozen=True, slots=True)
class TowerRefinement:
    schema_version: str
    status: TowerRefinementStatus
    quality: TowerRefinementQuality
    evidence: Mapping[str, Any]
    algorithm_version: str
    confirmed_at: str
    local_region_reference: ArtifactReference | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_REFINEMENT_SCHEMA_VERSION:
            raise ValueError("unsupported tower-refinement schema version")
        _enum(self.status, TowerRefinementStatus, "status")
        _enum(self.quality, TowerRefinementQuality, "quality")
        object.__setattr__(self, "evidence", _immutable_metadata(self.evidence, "evidence"))
        _exact_string(self.algorithm_version, "algorithm_version")
        _timestamp(self.confirmed_at, "confirmed_at")
        if (
            self.local_region_reference is not None
            and type(self.local_region_reference) is not ArtifactReference
        ):
            raise ValueError("local_region_reference must be an ArtifactReference or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "quality": self.quality.value,
            "evidence": _thaw_metadata(self.evidence),
            "algorithm_version": self.algorithm_version,
            "confirmed_at": self.confirmed_at,
            "local_region_reference": (
                self.local_region_reference.to_dict()
                if self.local_region_reference is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class TowerContract:
    """One confirmed physical tower; spans only retain an ID/revision reference to it."""

    schema_version: str
    tower_id: str
    display_name: str
    state: TowerLifecycleState
    source_binding: SourceIdentityBinding
    coordinate_context: CoordinateContext
    approximate_click_xyz_absolute_m: tuple[float, float, float]
    refined_geometry: TowerGeometry
    refinement: TowerRefinement
    revision: int
    notes: str | None
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_SCHEMA_VERSION:
            raise ValueError("unsupported tower schema version")
        _display_id(self.tower_id, _TOWER_ID_RE, "T", "tower_id")
        _exact_string(self.display_name, "display_name")
        _enum(self.state, TowerLifecycleState, "state")
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        if self.coordinate_context != self.source_binding.coordinate_context:
            raise ValueError("coordinate_context must match source_binding")
        object.__setattr__(
            self,
            "approximate_click_xyz_absolute_m",
            _xyz(self.approximate_click_xyz_absolute_m, "approximate_click_xyz_absolute_m"),
        )
        if type(self.refined_geometry) is not TowerGeometry:
            raise ValueError("refined_geometry must be a TowerGeometry")
        if type(self.refinement) is not TowerRefinement:
            raise ValueError("refinement must be a TowerRefinement")
        _positive_int(self.revision, "revision")
        if self.notes is not None:
            _exact_string(self.notes, "notes")
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if datetime.fromisoformat(updated[:-1] + "+00:00") < datetime.fromisoformat(
            created[:-1] + "+00:00"
        ):
            raise ValueError("updated_at must not precede created_at")

    @property
    def revision_reference(self) -> TowerRevisionReference:
        return TowerRevisionReference(tower_id=self.tower_id, revision=self.revision)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tower_id": self.tower_id,
            "display_name": self.display_name,
            "state": self.state.value,
            "source_binding": self.source_binding.to_dict(),
            "coordinate_context": self.coordinate_context.to_dict(),
            "approximate_click_xyz_absolute_m": list(self.approximate_click_xyz_absolute_m),
            "refined_geometry": self.refined_geometry.to_dict(),
            "refinement": self.refinement.to_dict(),
            "revision": self.revision,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class TowerRevisionReference:
    tower_id: str
    revision: int

    def __post_init__(self) -> None:
        _display_id(self.tower_id, _TOWER_ID_RE, "T", "tower_id")
        _positive_int(self.revision, "revision")

    def to_dict(self) -> dict[str, Any]:
        return {"tower_id": self.tower_id, "revision": self.revision}


@dataclass(frozen=True, slots=True)
class SpanGeometry:
    schema_version: str
    definition_revision: int
    coordinate_context: CoordinateContext
    length_m: float
    direction_xy: tuple[float, float]
    core_bounds_min_xyz: tuple[float, float, float]
    core_bounds_max_xyz: tuple[float, float, float]
    halo_bounds_min_xyz: tuple[float, float, float]
    halo_bounds_max_xyz: tuple[float, float, float]
    configuration_digest: str
    geometry_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != SPAN_GEOMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported span-geometry schema version")
        _positive_int(self.definition_revision, "definition_revision")
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        length = _finite_number(self.length_m, "length_m")
        if length <= 0.0:
            raise ValueError("length_m must be positive")
        direction = _vector(self.direction_xy, 2, "direction_xy")
        magnitude = math.hypot(direction[0], direction[1])
        if not math.isclose(magnitude, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("direction_xy must be a unit vector")
        mins_maxes = (
            (
                "core_bounds",
                _xyz(self.core_bounds_min_xyz, "core_bounds_min_xyz"),
                _xyz(self.core_bounds_max_xyz, "core_bounds_max_xyz"),
            ),
            (
                "halo_bounds",
                _xyz(self.halo_bounds_min_xyz, "halo_bounds_min_xyz"),
                _xyz(self.halo_bounds_max_xyz, "halo_bounds_max_xyz"),
            ),
        )
        for name, minimum, maximum in mins_maxes:
            if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
                raise ValueError(f"{name} minimum must not exceed maximum")
        core_minimum, core_maximum = mins_maxes[0][1], mins_maxes[0][2]
        halo_minimum, halo_maximum = mins_maxes[1][1], mins_maxes[1][2]
        if any(
            halo_lower > core_lower or core_upper > halo_upper
            for halo_lower, core_lower, core_upper, halo_upper in zip(
                halo_minimum,
                core_minimum,
                core_maximum,
                halo_maximum,
                strict=True,
            )
        ):
            raise ValueError("halo_bounds must contain core_bounds in every dimension")
        object.__setattr__(self, "length_m", length)
        object.__setattr__(self, "direction_xy", (direction[0], direction[1]))
        object.__setattr__(self, "core_bounds_min_xyz", mins_maxes[0][1])
        object.__setattr__(self, "core_bounds_max_xyz", mins_maxes[0][2])
        object.__setattr__(self, "halo_bounds_min_xyz", mins_maxes[1][1])
        object.__setattr__(self, "halo_bounds_max_xyz", mins_maxes[1][2])
        _sha256(self.configuration_digest, "configuration_digest")
        _sha256(self.geometry_digest, "geometry_digest")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "definition_revision": self.definition_revision,
            "coordinate_context": self.coordinate_context.to_dict(),
            "length_m": self.length_m,
            "direction_xy": list(self.direction_xy),
            "core_bounds_min_xyz": list(self.core_bounds_min_xyz),
            "core_bounds_max_xyz": list(self.core_bounds_max_xyz),
            "halo_bounds_min_xyz": list(self.halo_bounds_min_xyz),
            "halo_bounds_max_xyz": list(self.halo_bounds_max_xyz),
            "configuration_digest": self.configuration_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.to_digest_dict(), "geometry_digest": self.geometry_digest}


def derive_span_geometry_digest(
    *,
    source_binding: SourceIdentityBinding,
    from_tower: TowerRevisionReference,
    to_tower: TowerRevisionReference,
    geometry: SpanGeometry,
) -> str:
    """Derive a geometry digest that binds source, endpoint revisions, and config."""

    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    if (
        type(from_tower) is not TowerRevisionReference
        or type(to_tower) is not TowerRevisionReference
    ):
        raise ValueError("tower endpoints must be TowerRevisionReference instances")
    if type(geometry) is not SpanGeometry:
        raise ValueError("geometry must be a SpanGeometry")
    return frv_digest(
        {
            "source_binding": source_binding.to_dict(),
            "from_tower": from_tower.to_dict(),
            "to_tower": to_tower.to_dict(),
            "geometry": geometry.to_digest_dict(),
        }
    )


@dataclass(frozen=True, slots=True)
class SpanDependencySnapshot:
    """Definition identity retained by a run so changed topology can become stale."""

    source_binding: SourceIdentityBinding
    from_tower: TowerRevisionReference
    to_tower: TowerRevisionReference
    geometry_digest: str
    configuration_digest: str

    def __post_init__(self) -> None:
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        if (
            type(self.from_tower) is not TowerRevisionReference
            or type(self.to_tower) is not TowerRevisionReference
        ):
            raise ValueError("tower references must be TowerRevisionReference instances")
        if self.from_tower.tower_id == self.to_tower.tower_id:
            raise ValueError("a span dependency snapshot must not be a self-span")
        _sha256(self.geometry_digest, "geometry_digest")
        _sha256(self.configuration_digest, "configuration_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_binding": self.source_binding.to_dict(),
            "from_tower": self.from_tower.to_dict(),
            "to_tower": self.to_tower.to_dict(),
            "geometry_digest": self.geometry_digest,
            "configuration_digest": self.configuration_digest,
        }


def _stage_statuses(value: Any) -> Mapping[AnalysisStage, AnalysisStatus]:
    if not isinstance(value, Mapping):
        raise ValueError("stage_statuses must be a mapping")
    expected = set(AnalysisStage)
    if set(value) != expected:
        raise ValueError("stage_statuses must contain every required analysis stage exactly once")
    frozen: dict[AnalysisStage, AnalysisStatus] = {}
    for stage in AnalysisStage:
        frozen[stage] = _enum(value[stage], AnalysisStatus, f"stage_statuses[{stage.value}]")
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class SpanAnalysis:
    schema_version: str
    state: AnalysisStatus
    active_run_id: str | None
    stage_statuses: Mapping[AnalysisStage, AnalysisStatus]
    run_dependencies: SpanDependencySnapshot | None
    historical_run_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SPAN_ANALYSIS_SCHEMA_VERSION:
            raise ValueError("unsupported span-analysis schema version")
        _enum(self.state, AnalysisStatus, "state")
        if self.state is AnalysisStatus.NOT_STARTED:
            if self.active_run_id is not None:
                raise ValueError("NOT_STARTED analysis must not have an active_run_id")
            if self.run_dependencies is not None:
                raise ValueError("NOT_STARTED analysis must not have run_dependencies")
        elif self.active_run_id is None:
            if self.state not in {AnalysisStatus.NOT_STARTED, AnalysisStatus.READY}:
                raise ValueError("active_run_id is required once a run has started")
            if self.run_dependencies is not None:
                raise ValueError("run_dependencies require an active_run_id")
        else:
            _run_id(self.active_run_id, "active_run_id")
            if type(self.run_dependencies) is not SpanDependencySnapshot:
                raise ValueError("active_run_id requires a SpanDependencySnapshot")
        statuses = _stage_statuses(self.stage_statuses)
        object.__setattr__(self, "stage_statuses", statuses)
        historical_run_ids = tuple(self.historical_run_ids)
        for run_id in historical_run_ids:
            _run_id(run_id, "historical_run_ids[]")
        if len(historical_run_ids) != len(set(historical_run_ids)):
            raise ValueError("historical_run_ids must not contain duplicates")
        if (
            self.state is AnalysisStatus.STALE
            and self.active_run_id not in historical_run_ids
        ):
            raise ValueError("STALE analysis must retain active_run_id in historical_run_ids")
        object.__setattr__(self, "historical_run_ids", historical_run_ids)
        stage_values = tuple(statuses.values())
        if self.state is AnalysisStatus.NOT_STARTED and any(
            status is not AnalysisStatus.NOT_STARTED for status in stage_values
        ):
            raise ValueError("NOT_STARTED analysis requires all stages to be NOT_STARTED")
        if self.state is AnalysisStatus.COMPLETE and any(
            status is not AnalysisStatus.COMPLETE for status in stage_values
        ):
            raise ValueError("COMPLETE analysis requires all stages to be COMPLETE")
        if self.state is AnalysisStatus.STALE and any(
            status is not AnalysisStatus.STALE for status in stage_values
        ):
            raise ValueError("STALE analysis requires all stages to be STALE")
        if self.state is AnalysisStatus.RUNNING and AnalysisStatus.RUNNING not in stage_values:
            raise ValueError("RUNNING analysis requires a RUNNING stage")
        if self.state is AnalysisStatus.FAILED and AnalysisStatus.FAILED not in stage_values:
            raise ValueError("FAILED analysis requires a FAILED stage")
        if (
            self.state is AnalysisStatus.INTERRUPTED
            and AnalysisStatus.INTERRUPTED not in stage_values
        ):
            raise ValueError("INTERRUPTED analysis requires an INTERRUPTED stage")

    @classmethod
    def not_started(cls) -> SpanAnalysis:
        return cls(
            schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
            state=AnalysisStatus.NOT_STARTED,
            active_run_id=None,
            stage_statuses={stage: AnalysisStatus.NOT_STARTED for stage in AnalysisStage},
            run_dependencies=None,
            historical_run_ids=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.value,
            "active_run_id": self.active_run_id,
            "stage_statuses": {
                stage.value: self.stage_statuses[stage].value for stage in AnalysisStage
            },
            "run_dependencies": (
                self.run_dependencies.to_dict() if self.run_dependencies is not None else None
            ),
            "historical_run_ids": list(self.historical_run_ids),
        }


@dataclass(frozen=True, slots=True)
class ProviderPayload:
    """Versioned provider-owned payload boundary; core never interprets its contents."""

    namespace: str
    schema_version: str
    data: Mapping[str, Any]

    def __post_init__(self) -> None:
        namespace = _exact_string(self.namespace, "namespace")
        if _PROVIDER_ID_RE.fullmatch(namespace) is None:
            raise ValueError("namespace must be a lower-case namespaced provider identifier")
        _exact_string(self.schema_version, "schema_version")
        object.__setattr__(self, "data", _immutable_metadata(self.data, "data"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "schema_version": self.schema_version,
            "data": _thaw_metadata(self.data),
        }


@dataclass(frozen=True, slots=True)
class PredictionCapability:
    schema_version: str
    status: PredictionCapabilityState
    provider_id: str | None
    provider_version: str | None
    supported_input_schema_versions: tuple[str, ...]
    supported_result_schema_versions: tuple[str, ...]
    supported_execution_modes: tuple[PredictionExecutionMode, ...]
    diagnostic: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-capability schema version")
        _enum(self.status, PredictionCapabilityState, "status")
        input_versions = _ordered_unique_strings(
            self.supported_input_schema_versions, "supported_input_schema_versions"
        )
        result_versions = _ordered_unique_strings(
            self.supported_result_schema_versions, "supported_result_schema_versions"
        )
        modes = _ordered_unique_enums(
            self.supported_execution_modes,
            PredictionExecutionMode,
            "supported_execution_modes",
        )
        object.__setattr__(self, "supported_input_schema_versions", input_versions)
        object.__setattr__(self, "supported_result_schema_versions", result_versions)
        object.__setattr__(self, "supported_execution_modes", modes)
        if self.diagnostic is not None:
            _exact_string(self.diagnostic, "diagnostic")
        if self.status is PredictionCapabilityState.NOT_CONNECTED:
            if (
                self.provider_id is not None
                or self.provider_version is not None
                or input_versions
                or result_versions
                or modes
            ):
                raise ValueError("NOT_CONNECTED capability must not declare a provider")
            return
        provider_id = _exact_string(self.provider_id, "provider_id")
        if _PROVIDER_ID_RE.fullmatch(provider_id) is None:
            raise ValueError("provider_id must be a lower-case namespaced provider identifier")
        _exact_string(self.provider_version, "provider_version")
        if not input_versions or not result_versions or not modes:
            raise ValueError("connected prediction capability requires schemas and execution modes")

    @classmethod
    def not_connected(cls) -> PredictionCapability:
        return cls(
            schema_version=PREDICTION_CAPABILITY_SCHEMA_VERSION,
            status=PredictionCapabilityState.NOT_CONNECTED,
            provider_id=None,
            provider_version=None,
            supported_input_schema_versions=(),
            supported_result_schema_versions=(),
            supported_execution_modes=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "supported_input_schema_versions": list(self.supported_input_schema_versions),
            "supported_result_schema_versions": list(self.supported_result_schema_versions),
            "supported_execution_modes": [mode.value for mode in self.supported_execution_modes],
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True, slots=True)
class PredictionProviderDescriptor:
    schema_version: str
    provider_id: str
    provider_version: str
    supported_input_schema_versions: tuple[str, ...]
    supported_result_schema_versions: tuple[str, ...]
    supported_execution_modes: tuple[PredictionExecutionMode, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_PROVIDER_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-provider schema version")
        provider_id = _exact_string(self.provider_id, "provider_id")
        if _PROVIDER_ID_RE.fullmatch(provider_id) is None:
            raise ValueError("provider_id must be a lower-case namespaced provider identifier")
        _exact_string(self.provider_version, "provider_version")
        inputs = _ordered_unique_strings(
            self.supported_input_schema_versions, "supported_input_schema_versions"
        )
        results = _ordered_unique_strings(
            self.supported_result_schema_versions, "supported_result_schema_versions"
        )
        modes = _ordered_unique_enums(
            self.supported_execution_modes,
            PredictionExecutionMode,
            "supported_execution_modes",
        )
        if not inputs or not results or not modes:
            raise ValueError("provider descriptor requires schemas and execution modes")
        object.__setattr__(self, "supported_input_schema_versions", inputs)
        object.__setattr__(self, "supported_result_schema_versions", results)
        object.__setattr__(self, "supported_execution_modes", modes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "supported_input_schema_versions": list(self.supported_input_schema_versions),
            "supported_result_schema_versions": list(self.supported_result_schema_versions),
            "supported_execution_modes": [mode.value for mode in self.supported_execution_modes],
        }


@dataclass(frozen=True, slots=True)
class PredictionArtifactReference:
    schema_version: str
    relative_path: str
    role: str
    sha256: str
    byte_size: int
    artifact_schema_version: str

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-artifact-reference schema version")
        _safe_relative_posix_path(self.relative_path, "relative_path")
        _exact_string(self.role, "role")
        _sha256(self.sha256, "sha256")
        _nonnegative_int(self.byte_size, "byte_size")
        _exact_string(self.artifact_schema_version, "artifact_schema_version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relative_path": self.relative_path,
            "role": self.role,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "artifact_schema_version": self.artifact_schema_version,
        }


@dataclass(frozen=True, slots=True)
class AnalysisTreeObservationIdentity:
    """An analysis/run-local observation, explicitly not a permanent biological tree."""

    analysis_run_id: str
    local_tree_id: int
    observation_id: str

    def __post_init__(self) -> None:
        run_id = _run_id(self.analysis_run_id, "analysis_run_id")
        local_tree_id = _positive_int(self.local_tree_id, "local_tree_id")
        expected = f"{run_id}:tree:{local_tree_id}"
        if self.observation_id != expected:
            raise ValueError(
                "observation_id must be derived from analysis_run_id and local_tree_id"
            )

    @classmethod
    def create(cls, analysis_run_id: str, local_tree_id: int) -> AnalysisTreeObservationIdentity:
        run_id = _run_id(analysis_run_id, "analysis_run_id")
        tree_id = _positive_int(local_tree_id, "local_tree_id")
        return cls(
            analysis_run_id=run_id,
            local_tree_id=tree_id,
            observation_id=f"{run_id}:tree:{tree_id}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_run_id": self.analysis_run_id,
            "local_tree_id": self.local_tree_id,
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    schema_version: str
    request_id: str
    project_id: str
    source_binding: SourceIdentityBinding
    span_id: str
    span_dependencies: SpanDependencySnapshot
    analysis_run_id: str
    static_result_schema_version: str
    coordinate_context: CoordinateContext
    provider: PredictionProviderDescriptor
    tree_observations: tuple[AnalysisTreeObservationIdentity, ...]
    static_artifacts: tuple[PredictionArtifactReference, ...]
    provider_payload: ProviderPayload
    requested_at: str

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-request schema version")
        _opaque_id(self.request_id, "request_id")
        _project_id(self.project_id)
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        _display_id(self.span_id, _SPAN_ID_RE, "S", "span_id")
        if type(self.span_dependencies) is not SpanDependencySnapshot:
            raise ValueError("span_dependencies must be a SpanDependencySnapshot")
        if self.source_binding != self.span_dependencies.source_binding:
            raise ValueError("source_binding must match span_dependencies")
        run_id = _run_id(self.analysis_run_id, "analysis_run_id")
        _exact_string(self.static_result_schema_version, "static_result_schema_version")
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        if self.coordinate_context != self.source_binding.coordinate_context:
            raise ValueError("coordinate_context must match source_binding")
        if type(self.provider) is not PredictionProviderDescriptor:
            raise ValueError("provider must be a PredictionProviderDescriptor")
        if self.schema_version not in self.provider.supported_input_schema_versions:
            raise ValueError("provider does not support the prediction request schema")
        observations = tuple(self.tree_observations)
        if len({item.observation_id for item in observations}) != len(observations):
            raise ValueError("tree_observations must be unique by observation_id")
        for observation in observations:
            if type(observation) is not AnalysisTreeObservationIdentity:
                raise ValueError("tree_observations must contain AnalysisTreeObservationIdentity")
            if observation.analysis_run_id != run_id:
                raise ValueError("tree observations must belong to analysis_run_id")
        artifacts = tuple(self.static_artifacts)
        if any(type(item) is not PredictionArtifactReference for item in artifacts):
            raise ValueError("static_artifacts must contain PredictionArtifactReference")
        if type(self.provider_payload) is not ProviderPayload:
            raise ValueError("provider_payload must be a ProviderPayload")
        _timestamp(self.requested_at, "requested_at")
        object.__setattr__(self, "tree_observations", observations)
        object.__setattr__(self, "static_artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "project_id": self.project_id,
            "source_binding": self.source_binding.to_dict(),
            "span_id": self.span_id,
            "span_dependencies": self.span_dependencies.to_dict(),
            "analysis_run_id": self.analysis_run_id,
            "static_result_schema_version": self.static_result_schema_version,
            "coordinate_context": self.coordinate_context.to_dict(),
            "provider": self.provider.to_dict(),
            "tree_observations": [item.to_dict() for item in self.tree_observations],
            "static_artifacts": [item.to_dict() for item in self.static_artifacts],
            "provider_payload": self.provider_payload.to_dict(),
            "requested_at": self.requested_at,
        }


@dataclass(frozen=True, slots=True)
class PredictionResultEntry:
    observation: AnalysisTreeObservationIdentity
    provider_payload: ProviderPayload
    artifacts: tuple[PredictionArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if type(self.observation) is not AnalysisTreeObservationIdentity:
            raise ValueError("observation must be an AnalysisTreeObservationIdentity")
        if type(self.provider_payload) is not ProviderPayload:
            raise ValueError("provider_payload must be a ProviderPayload")
        artifacts = tuple(self.artifacts)
        if any(type(item) is not PredictionArtifactReference for item in artifacts):
            raise ValueError("artifacts must contain PredictionArtifactReference")
        object.__setattr__(self, "artifacts", artifacts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "provider_payload": self.provider_payload.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class PredictionResult:
    schema_version: str
    result_id: str
    request_id: str
    request_schema_version: str
    project_id: str
    source_binding: SourceIdentityBinding
    span_id: str
    span_dependencies: SpanDependencySnapshot
    analysis_run_id: str
    static_result_schema_version: str
    coordinate_context: CoordinateContext
    provider_id: str
    provider_version: str
    model_version: str
    entries: Mapping[str, PredictionResultEntry]
    artifacts: tuple[PredictionArtifactReference, ...]
    provider_payload: ProviderPayload
    completed_at: str

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported prediction-result schema version")
        _opaque_id(self.result_id, "result_id")
        _opaque_id(self.request_id, "request_id")
        if self.request_schema_version != PREDICTION_REQUEST_SCHEMA_VERSION:
            raise ValueError(
                "request_schema_version must be the supported prediction request schema"
            )
        _project_id(self.project_id)
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        _display_id(self.span_id, _SPAN_ID_RE, "S", "span_id")
        if type(self.span_dependencies) is not SpanDependencySnapshot:
            raise ValueError("span_dependencies must be a SpanDependencySnapshot")
        if self.source_binding != self.span_dependencies.source_binding:
            raise ValueError("source_binding must match span_dependencies")
        _run_id(self.analysis_run_id, "analysis_run_id")
        _exact_string(self.static_result_schema_version, "static_result_schema_version")
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        if self.coordinate_context != self.source_binding.coordinate_context:
            raise ValueError("coordinate_context must match source_binding")
        provider_id = _exact_string(self.provider_id, "provider_id")
        if _PROVIDER_ID_RE.fullmatch(provider_id) is None:
            raise ValueError("provider_id must be a lower-case namespaced provider identifier")
        _exact_string(self.provider_version, "provider_version")
        _exact_string(self.model_version, "model_version")
        if not isinstance(self.entries, Mapping):
            raise ValueError("entries must be an identity-keyed mapping")
        entries: dict[str, PredictionResultEntry] = {}
        for observation_id, entry in self.entries.items():
            key = _exact_string(observation_id, "entries key")
            if type(entry) is not PredictionResultEntry or entry.observation.observation_id != key:
                raise ValueError("entries must be keyed by their observation_id")
            if entry.observation.analysis_run_id != self.analysis_run_id:
                raise ValueError("result entries must belong to analysis_run_id")
            entries[key] = entry
        object.__setattr__(self, "entries", MappingProxyType(dict(entries)))
        artifacts = tuple(self.artifacts)
        if any(type(item) is not PredictionArtifactReference for item in artifacts):
            raise ValueError("artifacts must contain PredictionArtifactReference")
        object.__setattr__(self, "artifacts", artifacts)
        if type(self.provider_payload) is not ProviderPayload:
            raise ValueError("provider_payload must be a ProviderPayload")
        _timestamp(self.completed_at, "completed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "request_id": self.request_id,
            "request_schema_version": self.request_schema_version,
            "project_id": self.project_id,
            "source_binding": self.source_binding.to_dict(),
            "span_id": self.span_id,
            "span_dependencies": self.span_dependencies.to_dict(),
            "analysis_run_id": self.analysis_run_id,
            "static_result_schema_version": self.static_result_schema_version,
            "coordinate_context": self.coordinate_context.to_dict(),
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "model_version": self.model_version,
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
            "artifacts": [item.to_dict() for item in self.artifacts],
            "provider_payload": self.provider_payload.to_dict(),
            "completed_at": self.completed_at,
        }


def validate_prediction_result_for_request(
    request: PredictionRequest, result: PredictionResult
) -> None:
    """Verify a result's stable identity binding to one immutable request."""

    if type(request) is not PredictionRequest or type(result) is not PredictionResult:
        raise ValueError("request and result must use their exact P1 contract types")
    bindings = (
        ("request_id", result.request_id, request.request_id),
        ("project_id", result.project_id, request.project_id),
        ("source_binding", result.source_binding, request.source_binding),
        ("span_id", result.span_id, request.span_id),
        ("span_dependencies", result.span_dependencies, request.span_dependencies),
        ("analysis_run_id", result.analysis_run_id, request.analysis_run_id),
        (
            "static_result_schema_version",
            result.static_result_schema_version,
            request.static_result_schema_version,
        ),
        ("coordinate_context", result.coordinate_context, request.coordinate_context),
        ("provider_id", result.provider_id, request.provider.provider_id),
        ("provider_version", result.provider_version, request.provider.provider_version),
    )
    for name, actual, expected in bindings:
        if actual != expected:
            raise ValueError(f"prediction result {name} does not match request")
    if result.schema_version not in request.provider.supported_result_schema_versions:
        raise ValueError("provider does not support the prediction result schema")
    requested_observations = {item.observation_id for item in request.tree_observations}
    if not set(result.entries).issubset(requested_observations):
        raise ValueError("prediction result entries must be requested analysis-scoped observations")


@dataclass(frozen=True, slots=True)
class SpanContract:
    """Project-level span topology, status, and current/historical dependency semantics."""

    schema_version: str
    span_id: str
    display_name: str
    from_tower_id: str
    to_tower_id: str
    from_tower_revision: int
    to_tower_revision: int
    source_binding: SourceIdentityBinding
    topology_state: SpanTopologyState
    geometry: SpanGeometry
    analysis: SpanAnalysis
    artifact_index_reference: ArtifactReference | None
    prediction: PredictionCapability = field(default_factory=PredictionCapability.not_connected)

    def __post_init__(self) -> None:
        if self.schema_version != SPAN_SCHEMA_VERSION:
            raise ValueError("unsupported span schema version")
        _display_id(self.span_id, _SPAN_ID_RE, "S", "span_id")
        _exact_string(self.display_name, "display_name")
        _display_id(self.from_tower_id, _TOWER_ID_RE, "T", "from_tower_id")
        _display_id(self.to_tower_id, _TOWER_ID_RE, "T", "to_tower_id")
        if self.from_tower_id == self.to_tower_id:
            raise ValueError("project spans must not use the same tower for both endpoints")
        _positive_int(self.from_tower_revision, "from_tower_revision")
        _positive_int(self.to_tower_revision, "to_tower_revision")
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        _enum(self.topology_state, SpanTopologyState, "topology_state")
        if type(self.geometry) is not SpanGeometry:
            raise ValueError("geometry must be a SpanGeometry")
        if self.geometry.coordinate_context != self.source_binding.coordinate_context:
            raise ValueError("geometry coordinate_context must match source_binding")
        expected_digest = derive_span_geometry_digest(
            source_binding=self.source_binding,
            from_tower=self.from_tower_reference,
            to_tower=self.to_tower_reference,
            geometry=self.geometry,
        )
        if self.geometry.geometry_digest != expected_digest:
            raise ValueError(
                "geometry_digest does not bind span source, endpoint revisions, and config"
            )
        if type(self.analysis) is not SpanAnalysis:
            raise ValueError("analysis must be a SpanAnalysis")
        if self.analysis.run_dependencies is not None:
            stale = self.analysis.run_dependencies != self.current_dependencies
            if stale != (self.analysis.state is AnalysisStatus.STALE):
                raise ValueError(
                    "analysis state must be STALE exactly when run dependencies differ"
                )
        if (
            self.artifact_index_reference is not None
            and type(self.artifact_index_reference) is not ArtifactReference
        ):
            raise ValueError("artifact_index_reference must be an ArtifactReference or null")
        if type(self.prediction) is not PredictionCapability:
            raise ValueError("prediction must be a PredictionCapability")

    @property
    def from_tower_reference(self) -> TowerRevisionReference:
        return TowerRevisionReference(self.from_tower_id, self.from_tower_revision)

    @property
    def to_tower_reference(self) -> TowerRevisionReference:
        return TowerRevisionReference(self.to_tower_id, self.to_tower_revision)

    @property
    def current_dependencies(self) -> SpanDependencySnapshot:
        return SpanDependencySnapshot(
            source_binding=self.source_binding,
            from_tower=self.from_tower_reference,
            to_tower=self.to_tower_reference,
            geometry_digest=self.geometry.geometry_digest,
            configuration_digest=self.geometry.configuration_digest,
        )

    @property
    def is_stale(self) -> bool:
        return self.analysis.run_dependencies is not None and (
            self.analysis.run_dependencies != self.current_dependencies
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "span_id": self.span_id,
            "display_name": self.display_name,
            "from_tower_id": self.from_tower_id,
            "to_tower_id": self.to_tower_id,
            "from_tower_revision": self.from_tower_revision,
            "to_tower_revision": self.to_tower_revision,
            "source_binding": self.source_binding.to_dict(),
            "topology_state": self.topology_state.value,
            "geometry": self.geometry.to_dict(),
            "analysis": self.analysis.to_dict(),
            "artifact_index_reference": (
                self.artifact_index_reference.to_dict()
                if self.artifact_index_reference is not None
                else None
            ),
            "prediction": self.prediction.to_dict(),
        }


def _project_id(value: Any) -> str:
    text = _exact_string(value, "project_id")
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("project_id must be a canonical UUID") from exc
    if str(parsed) != text:
        raise ValueError("project_id must be a canonical lower-case UUID")
    return text


@dataclass(frozen=True, slots=True)
class ProjectContract:
    """P1 project aggregate contract. Registry fields are references, not persistence."""

    schema_version: str
    project_id: str
    project_name: str
    created_at: str
    updated_at: str
    source_reference: ProjectSourceReference
    viewer: ViewerReference
    tower_registry_reference: str
    span_registry_reference: str
    run_index_reference: str
    next_tower_sequence: int
    next_span_sequence: int
    prediction_capability: PredictionCapability = field(
        default_factory=PredictionCapability.not_connected
    )

    def __post_init__(self) -> None:
        if self.schema_version != PROJECT_SCHEMA_VERSION:
            raise ValueError("unsupported project schema version")
        _project_id(self.project_id)
        _exact_string(self.project_name, "project_name")
        created = _timestamp(self.created_at, "created_at")
        updated = _timestamp(self.updated_at, "updated_at")
        if datetime.fromisoformat(updated[:-1] + "+00:00") < datetime.fromisoformat(
            created[:-1] + "+00:00"
        ):
            raise ValueError("updated_at must not precede created_at")
        if type(self.source_reference) is not ProjectSourceReference:
            raise ValueError("source_reference must be a ProjectSourceReference")
        if type(self.viewer) is not ViewerReference:
            raise ValueError("viewer must be a ViewerReference")
        _safe_relative_posix_path(self.tower_registry_reference, "tower_registry_reference")
        _safe_relative_posix_path(self.span_registry_reference, "span_registry_reference")
        _safe_relative_posix_path(self.run_index_reference, "run_index_reference")
        _positive_int(self.next_tower_sequence, "next_tower_sequence")
        _positive_int(self.next_span_sequence, "next_span_sequence")
        if type(self.prediction_capability) is not PredictionCapability:
            raise ValueError("prediction_capability must be a PredictionCapability")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_reference": self.source_reference.to_dict(),
            "viewer": self.viewer.to_dict(),
            "tower_registry_reference": self.tower_registry_reference,
            "span_registry_reference": self.span_registry_reference,
            "run_index_reference": self.run_index_reference,
            "prediction_capability": self.prediction_capability.to_dict(),
            "next_tower_sequence": self.next_tower_sequence,
            "next_span_sequence": self.next_span_sequence,
        }


__all__ = [
    "PROJECT_SCHEMA_VERSION",
    "PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION",
    "COORDINATE_CONTEXT_SCHEMA_VERSION",
    "TOWER_SCHEMA_VERSION",
    "TOWER_REFINEMENT_SCHEMA_VERSION",
    "SPAN_SCHEMA_VERSION",
    "SPAN_GEOMETRY_SCHEMA_VERSION",
    "SPAN_ANALYSIS_SCHEMA_VERSION",
    "VIEWER_REFERENCE_SCHEMA_VERSION",
    "PREDICTION_CAPABILITY_SCHEMA_VERSION",
    "PREDICTION_PROVIDER_SCHEMA_VERSION",
    "PREDICTION_REQUEST_SCHEMA_VERSION",
    "PREDICTION_RESULT_SCHEMA_VERSION",
    "PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION",
    "CoordinateFrame",
    "CoordinateUnit",
    "TowerLifecycleState",
    "TowerRefinementStatus",
    "TowerRefinementQuality",
    "SpanTopologyState",
    "AnalysisStatus",
    "AnalysisStage",
    "ViewerCacheState",
    "PredictionCapabilityState",
    "PredictionExecutionMode",
    "CoordinateContext",
    "SourceIdentityBinding",
    "ProjectSourceReference",
    "ViewerReference",
    "ArtifactReference",
    "TowerGeometry",
    "TowerRefinement",
    "TowerContract",
    "TowerRevisionReference",
    "SpanGeometry",
    "SpanDependencySnapshot",
    "SpanAnalysis",
    "SpanContract",
    "ProviderPayload",
    "PredictionCapability",
    "PredictionProviderDescriptor",
    "PredictionArtifactReference",
    "AnalysisTreeObservationIdentity",
    "PredictionRequest",
    "PredictionResultEntry",
    "PredictionResult",
    "derive_span_geometry_digest",
    "tower_display_id_for_sequence",
    "span_display_id_for_sequence",
    "validate_prediction_result_for_request",
]
