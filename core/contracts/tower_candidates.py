"""Immutable V5A1-A tower candidate / review contracts.

These contracts freeze the proposal, candidate-set, detector-provenance,
review-event, review-ledger, and candidate-confirmation boundaries for the
tower-candidate review pipeline.  They deliberately do *not* implement
automatic tower detection, a wire-topology detector, a Giro3D review UI, the
RAW refinement algorithm, project persistence v2, review-to-TowerRegistry
writes, ``create_tower`` integration, span ordering, or span creation.

Candidate IDs are the ``P###`` display domain, scoped by the identity of one
``TowerCandidateSet`` (candidate_set_id + candidate_set_digest + candidate_id).
Confirmed tower IDs (``T###``) remain owned by the existing Tower Registry and
are never allocated here.  No ``T###`` may ever be interpreted as a candidate
ID and no ``P###`` may ever be interpreted as a tower ID.

The candidate set is immutable and deterministic.  A reproducible rerun with
identical candidate contents is still a *new* generation with a new
``candidate_set_id``; its content digest (which excludes the set id, the
generation timestamp, and the digest field itself) is a pure fingerprint of
the scientific content.

Review history is separate from the candidate set: an immutable, append-only
ledger of semantic review events.  Accepted reviews that may later allocate a
Tower must carry RAW_REFINED defensible geometry/evidence.  Rejected and
superseded observations can never become confirmations.  Current/effective
state is projected from the immutable ledger; there is no mutable "current
review state" source of truth.

This module performs no file access, no persistence, no detection, and no
refinement.  It only validates and serializes the frozen contracts.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntFlag
from types import MappingProxyType
from typing import Any
from uuid import UUID

from core.contracts._json import json_safe_mapping, reject_unknown_fields, require_fields
from core.contracts.frv import frv_digest
from core.contracts.project import (
    CoordinateContext,
    CoordinateFrame,
    CoordinateUnit,
    SourceIdentityBinding,
    TowerGeometry,
    TowerRevisionReference,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

TOWER_CANDIDATE_SET_SCHEMA_VERSION = "treeline-tower-candidate-set-v1"
TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION = "treeline-tower-candidate-review-event-v1"
TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION = "treeline-tower-candidate-review-ledger-v1"
TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION = "treeline-tower-candidate-confirmation-v1"

# Declared digest domain identifier for the tower-candidate-set content digest.
TOWER_CANDIDATE_SET_DIGEST_DOMAIN = "TREELINETCSET1"

# ---------------------------------------------------------------------------
# Closed enums and quality flags
# ---------------------------------------------------------------------------


class CandidateOrigin(str, Enum):
    """Source of a tower candidate.  ``MANUAL_ADD`` is an origin, never a review decision."""

    AUTOMATIC = "AUTOMATIC"
    MANUAL_ADD = "MANUAL_ADD"


class DetectorKind(str, Enum):
    """Closed detector taxonomy.  Future detectors must stay inside these kinds."""

    GEOMETRY = "GEOMETRY"
    WIRE_TOPOLOGY = "WIRE_TOPOLOGY"
    ML = "ML"
    HYBRID = "HYBRID"


class RefinementState(str, Enum):
    """RAW refinement state of a candidate's proposal geometry."""

    UNREFINED = "UNREFINED"
    RAW_REFINED = "RAW_REFINED"
    RAW_REFINEMENT_FAILED = "RAW_REFINEMENT_FAILED"


class ReviewDecision(str, Enum):
    """Semantic review decision carried by one immutable review event."""

    PENDING_REVIEW = "PENDING_REVIEW"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class TowerCandidateQualityFlag(IntFlag):
    """Closed per-candidate quality bits.  Unknown bits are always rejected."""

    NONE = 0
    GEOMETRY_QUALITY_DEGRADED = 1
    GROUND_CONTEXT_UNCERTAIN = 2
    OCCLUSION_SUSPECTED = 4
    LOW_STRUCTURAL_CONFIDENCE = 8
    AMBIGUOUS_CONTEXT = 16
    PROPOSAL_NEEDS_REFINEMENT = 32


# ---------------------------------------------------------------------------
# Private scalar/container validation helpers
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
# Frozen P### identity domain: literal P followed by exactly three decimal
# digits (P001..P999).  No extra leading zeros, no lowercase, no signs.
_CANDIDATE_ID_RE = re.compile(r"P([0-9]{3})\Z")
_OPAQUE_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_MAX_METADATA_NODES = 512
_MAX_METADATA_SEQUENCE_LENGTH = 128


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


def _finite_unit_interval(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1]")
    return result


def _enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _enum_from_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be a boolean")
    if type(value) is not str:
        raise ValueError(f"{name} must be a string enum value")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


def _flag_mask(value: Any, flag_type: type[IntFlag], name: str) -> IntFlag:
    if isinstance(value, bool):
        raise ValueError(f"{name} must not be a boolean")
    if isinstance(value, flag_type):
        value = int(value)
    elif type(value) is not int:
        raise ValueError(f"{name} must be an integer flag mask")
    known = 0
    for member in flag_type.__members__.values():
        known |= int(member.value)
    if value < 0 or value > 0xFFFFFFFF or value & ~known:
        raise ValueError(f"{name} contains unknown flag bit(s)")
    return flag_type(value)


def _timestamp(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if not text.endswith("Z") or "T" not in text:
        raise ValueError(f"{name} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    return text


def _opaque_identifier(value: Any, name: str) -> str:
    text = _exact_string(value, name)
    if _OPAQUE_ID_RE.fullmatch(text) is None:
        raise ValueError(f"{name} must be a lower-case opaque identifier")
    return text


def _candidate_id(value: Any) -> str:
    text = _exact_string(value, "candidate_id")
    matched = _CANDIDATE_ID_RE.fullmatch(text)
    if matched is None:
        raise ValueError("candidate_id must be a canonical P### candidate ID such as P001")
    sequence = int(matched.group(1))
    if not 1 <= sequence <= 999:
        raise ValueError("candidate_id must be a canonical P001-P999 candidate ID")
    return text


def _project_id(value: Any) -> str:
    text = _exact_string(value, "project_id")
    try:
        parsed = UUID(text)
    except (ValueError, AttributeError) as exc:
        raise ValueError("project_id must be a canonical UUID") from exc
    if str(parsed) != text:
        raise ValueError("project_id must be a canonical lower-case UUID")
    return text


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
        return tuple(_freeze_metadata(item, f"{name}[]", node_count) for item in value)
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


def _strict_mapping(data: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return data


def _from_dict_values(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    values = _strict_mapping(data, cls.__name__)
    fields = set(cls.__dataclass_fields__)
    require_fields(values, fields, cls.__name__)
    reject_unknown_fields(values, fields, cls.__name__)
    return dict(values)


def _json_array(value: Any, name: str) -> list[Any]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return value


def _tuple_from_json(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) is not list:
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _strict_json_dumps(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite numeric literal {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("JSON must be UTF-8 encoded") from exc
    try:
        return json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("JSON is malformed") from exc


def _tower_geometry_from_dict(data: Any) -> TowerGeometry:
    fields = {"base_xyz_absolute_m", "top_xyz_absolute_m", "height_m"}
    values = _strict_mapping(data, "refined_raw_geometry")
    require_fields(values, fields, "refined_raw_geometry")
    reject_unknown_fields(values, fields, "refined_raw_geometry")
    return TowerGeometry(
        base_xyz_absolute_m=tuple(values["base_xyz_absolute_m"]),
        top_xyz_absolute_m=tuple(values["top_xyz_absolute_m"]),
        height_m=values["height_m"],
    )


def _tower_revision_from_dict(data: Any) -> TowerRevisionReference:
    fields = {"tower_id", "revision"}
    values = _strict_mapping(data, "confirmed_tower")
    require_fields(values, fields, "confirmed_tower")
    reject_unknown_fields(values, fields, "confirmed_tower")
    return TowerRevisionReference(
        tower_id=values["tower_id"],
        revision=values["revision"],
    )


def _coordinate_context_from_dict(data: Any) -> CoordinateContext:
    fields = {
        "schema_version",
        "frame",
        "units",
        "origin_xyz",
        "source_content_identity_digest",
        "source_identity_record_digest",
    }
    values = _strict_mapping(data, "coordinate_context")
    require_fields(values, fields, "coordinate_context")
    reject_unknown_fields(values, fields, "coordinate_context")
    return CoordinateContext(
        schema_version=values["schema_version"],
        frame=CoordinateFrame(values["frame"]),
        units=CoordinateUnit(values["units"]),
        origin_xyz=tuple(values["origin_xyz"]),
        source_content_identity_digest=values["source_content_identity_digest"],
        source_identity_record_digest=values["source_identity_record_digest"],
    )


def _source_binding_from_dict(data: Any) -> SourceIdentityBinding:
    fields = {
        "source_content_identity_digest",
        "source_identity_record_digest",
        "coordinate_context",
    }
    values = _strict_mapping(data, "source_binding")
    require_fields(values, fields, "source_binding")
    reject_unknown_fields(values, fields, "source_binding")
    return SourceIdentityBinding(
        source_content_identity_digest=values["source_content_identity_digest"],
        source_identity_record_digest=values["source_identity_record_digest"],
        coordinate_context=_coordinate_context_from_dict(values["coordinate_context"]),
    )


# ---------------------------------------------------------------------------
# Nested candidate structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectorProvenance:
    """Detector-generic provenance for one generation or one candidate."""

    detector_kind: DetectorKind
    detector_name: str
    detector_version: str
    configuration_digest: str
    detector_evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _enum(self.detector_kind, DetectorKind, "detector_kind")
        _exact_string(self.detector_name, "detector_name")
        _exact_string(self.detector_version, "detector_version")
        _sha256(self.configuration_digest, "configuration_digest")
        object.__setattr__(
            self,
            "detector_evidence",
            _immutable_metadata(self.detector_evidence, "detector_evidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector_kind": self.detector_kind.value,
            "detector_name": self.detector_name,
            "detector_version": self.detector_version,
            "configuration_digest": self.configuration_digest,
            "detector_evidence": _thaw_metadata(self.detector_evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorProvenance:
        values = _from_dict_values(cls, data)
        values["detector_kind"] = _enum_from_value(
            values["detector_kind"], DetectorKind, "detector_kind"
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TowerCandidateManualAddProvenance:
    """Operator provenance for a ``MANUAL_ADD`` candidate."""

    added_at: str
    reason: str
    operator_identity: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        _timestamp(self.added_at, "added_at")
        _exact_string(self.reason, "reason")
        if self.operator_identity is not None:
            _exact_string(self.operator_identity, "operator_identity")
        if self.note is not None:
            _exact_string(self.note, "note")

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_at": self.added_at,
            "reason": self.reason,
            "operator_identity": self.operator_identity,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateManualAddProvenance:
        values = _from_dict_values(cls, data)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TowerCandidateBounds:
    """Inclusive axis-aligned bounds in absolute metres."""

    min_xyz_absolute_m: tuple[float, float, float]
    max_xyz_absolute_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        minimum = _xyz(self.min_xyz_absolute_m, "min_xyz_absolute_m")
        maximum = _xyz(self.max_xyz_absolute_m, "max_xyz_absolute_m")
        if any(lower > upper for lower, upper in zip(minimum, maximum, strict=True)):
            raise ValueError("bounds minimum must not exceed maximum in any dimension")
        object.__setattr__(self, "min_xyz_absolute_m", minimum)
        object.__setattr__(self, "max_xyz_absolute_m", maximum)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_xyz_absolute_m": list(self.min_xyz_absolute_m),
            "max_xyz_absolute_m": list(self.max_xyz_absolute_m),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateBounds:
        values = _from_dict_values(cls, data)
        values["min_xyz_absolute_m"] = _tuple_from_json(
            values["min_xyz_absolute_m"], "min_xyz_absolute_m"
        )
        values["max_xyz_absolute_m"] = _tuple_from_json(
            values["max_xyz_absolute_m"], "max_xyz_absolute_m"
        )
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TowerCandidateRawRefinementEvidence:
    """Closed RAW refinement evidence for an accepted review.

    Uses the same JSON-safe evidence convention as the existing
    ``TowerRefinement`` contract while freezing the fields needed to prove
    how a RAW refinement was produced and to which authoritative source it
    is bound.  ``details`` is supplementary JSON-safe metadata and may be
    empty; the closed fields above it are the required evidence.
    """

    method: str
    algorithm_version: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _exact_string(self.method, "method")
        _exact_string(self.algorithm_version, "algorithm_version")
        _sha256(self.source_content_identity_digest, "source_content_identity_digest")
        _sha256(self.source_identity_record_digest, "source_identity_record_digest")
        object.__setattr__(self, "details", _immutable_metadata(self.details, "details"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "algorithm_version": self.algorithm_version,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "details": _thaw_metadata(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateRawRefinementEvidence:
        values = _from_dict_values(cls, data)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TowerCandidate:
    """Immutable proposal for one possible tower.  Proposal geometry never changes."""

    candidate_id: str
    origin: CandidateOrigin
    coordinate_frame: CoordinateFrame
    coordinate_unit: CoordinateUnit
    candidate_xyz_absolute_m: tuple[float, float, float]
    candidate_bounds_absolute_m: TowerCandidateBounds | None = None
    confidence: float = 1.0
    quality_flags: TowerCandidateQualityFlag = TowerCandidateQualityFlag.NONE
    detector_provenance: DetectorProvenance | None = None
    manual_add_provenance: TowerCandidateManualAddProvenance | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _candidate_id(self.candidate_id)
        _enum(self.origin, CandidateOrigin, "origin")
        _enum(self.coordinate_frame, CoordinateFrame, "coordinate_frame")
        _enum(self.coordinate_unit, CoordinateUnit, "coordinate_unit")
        if self.coordinate_frame is not CoordinateFrame.SOURCE_LAS_ABSOLUTE:
            raise ValueError("proposal coordinate frame must be SOURCE_LAS_ABSOLUTE")
        if self.coordinate_unit is not CoordinateUnit.METRE:
            raise ValueError("proposal coordinate unit must be metres")
        object.__setattr__(
            self,
            "candidate_xyz_absolute_m",
            _xyz(self.candidate_xyz_absolute_m, "candidate_xyz_absolute_m"),
        )
        if (
            self.candidate_bounds_absolute_m is not None
            and type(self.candidate_bounds_absolute_m) is not TowerCandidateBounds
        ):
            raise ValueError("candidate_bounds_absolute_m must be a TowerCandidateBounds or null")
        object.__setattr__(
            self, "confidence", _finite_unit_interval(self.confidence, "confidence")
        )
        object.__setattr__(
            self,
            "quality_flags",
            _flag_mask(self.quality_flags, TowerCandidateQualityFlag, "quality_flags"),
        )
        if (
            self.detector_provenance is not None
            and type(self.detector_provenance) is not DetectorProvenance
        ):
            raise ValueError("detector_provenance must be a DetectorProvenance or null")
        if (
            self.manual_add_provenance is not None
            and type(self.manual_add_provenance) is not TowerCandidateManualAddProvenance
        ):
            raise ValueError(
                "manual_add_provenance must be a TowerCandidateManualAddProvenance or null"
            )
        if self.origin is CandidateOrigin.MANUAL_ADD:
            if self.manual_add_provenance is None:
                raise ValueError("MANUAL_ADD candidates require manual_add_provenance")
            if self.detector_provenance is not None:
                raise ValueError("MANUAL_ADD candidates must not carry detector_provenance")
        else:
            if self.detector_provenance is None:
                raise ValueError("AUTOMATIC candidates require detector_provenance")
            if self.manual_add_provenance is not None:
                raise ValueError("AUTOMATIC candidates must not carry manual_add_provenance")
        object.__setattr__(self, "evidence", _immutable_metadata(self.evidence, "evidence"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "origin": self.origin.value,
            "coordinate_frame": self.coordinate_frame.value,
            "coordinate_unit": self.coordinate_unit.value,
            "candidate_xyz_absolute_m": list(self.candidate_xyz_absolute_m),
            "candidate_bounds_absolute_m": (
                self.candidate_bounds_absolute_m.to_dict()
                if self.candidate_bounds_absolute_m is not None
                else None
            ),
            "confidence": self.confidence,
            "quality_flags": int(self.quality_flags),
            "detector_provenance": (
                self.detector_provenance.to_dict()
                if self.detector_provenance is not None
                else None
            ),
            "manual_add_provenance": (
                self.manual_add_provenance.to_dict()
                if self.manual_add_provenance is not None
                else None
            ),
            "evidence": _thaw_metadata(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidate:
        values = _from_dict_values(cls, data)
        values["origin"] = _enum_from_value(values["origin"], CandidateOrigin, "origin")
        values["coordinate_frame"] = _enum_from_value(
            values["coordinate_frame"], CoordinateFrame, "coordinate_frame"
        )
        values["coordinate_unit"] = _enum_from_value(
            values["coordinate_unit"], CoordinateUnit, "coordinate_unit"
        )
        values["candidate_xyz_absolute_m"] = _tuple_from_json(
            values["candidate_xyz_absolute_m"], "candidate_xyz_absolute_m"
        )
        if values["candidate_bounds_absolute_m"] is not None:
            values["candidate_bounds_absolute_m"] = TowerCandidateBounds.from_dict(
                values["candidate_bounds_absolute_m"]
            )
        values["quality_flags"] = _flag_mask(
            values["quality_flags"], TowerCandidateQualityFlag, "quality_flags"
        )
        if values["detector_provenance"] is not None:
            values["detector_provenance"] = DetectorProvenance.from_dict(
                values["detector_provenance"]
            )
        if values["manual_add_provenance"] is not None:
            values["manual_add_provenance"] = TowerCandidateManualAddProvenance.from_dict(
                values["manual_add_provenance"]
            )
        return cls(**values)


# ---------------------------------------------------------------------------
# Candidate set identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSetReference:
    """Digest-bound reference to one immutable ``TowerCandidateSet`` generation."""

    candidate_set_id: str
    candidate_set_digest: str

    def __post_init__(self) -> None:
        _opaque_identifier(self.candidate_set_id, "candidate_set_id")
        _sha256(self.candidate_set_digest, "candidate_set_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_set_id": self.candidate_set_id,
            "candidate_set_digest": self.candidate_set_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidateSetReference:
        values = _from_dict_values(cls, data)
        return cls(**values)


def _canonical_candidate_order(
    candidates: Sequence[TowerCandidate],
) -> tuple[TowerCandidate, ...]:
    """Order candidates canonically by candidate ID before storage and digesting.

    With the frozen three-digit P### domain, numeric and canonical lexical
    ordering coincide; sorting by the canonical ID keeps the intent explicit.
    """

    return tuple(sorted(candidates, key=lambda candidate: candidate.candidate_id))


def _candidate_set_digest_payload(
    *,
    schema_version: str,
    project_id: str,
    source_binding: SourceIdentityBinding,
    coordinate_context: CoordinateContext,
    detector_provenance: DetectorProvenance,
    candidates: Sequence[TowerCandidate],
    corridor_bounds_absolute_m: TowerCandidateBounds | None,
) -> dict[str, Any]:
    """Deterministic scientific-content payload; excludes set id, generation time, and digest."""

    return {
        "digest_domain": TOWER_CANDIDATE_SET_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "project_id": project_id,
        "source_binding": source_binding.to_dict(),
        "coordinate_context": coordinate_context.to_dict(),
        "detector_provenance": detector_provenance.to_dict(),
        "candidates": [candidate.to_dict() for candidate in candidates],
        "candidate_count": len(candidates),
        "corridor_bounds_absolute_m": (
            corridor_bounds_absolute_m.to_dict()
            if corridor_bounds_absolute_m is not None
            else None
        ),
    }


def derive_tower_candidate_set_digest(
    *,
    schema_version: str,
    project_id: str,
    source_binding: SourceIdentityBinding,
    coordinate_context: CoordinateContext,
    detector_provenance: DetectorProvenance,
    candidates: Sequence[TowerCandidate],
    corridor_bounds_absolute_m: TowerCandidateBounds | None = None,
) -> str:
    """Return the deterministic self-excluding content digest for a candidate set."""

    if schema_version != TOWER_CANDIDATE_SET_SCHEMA_VERSION:
        raise ValueError("unsupported tower-candidate-set schema version")
    _project_id(project_id)
    if type(source_binding) is not SourceIdentityBinding:
        raise ValueError("source_binding must be a SourceIdentityBinding")
    if type(coordinate_context) is not CoordinateContext:
        raise ValueError("coordinate_context must be a CoordinateContext")
    if coordinate_context != source_binding.coordinate_context:
        raise ValueError("coordinate_context must match source_binding")
    if type(detector_provenance) is not DetectorProvenance:
        raise ValueError("detector_provenance must be a DetectorProvenance")
    if not isinstance(candidates, Sequence) or isinstance(candidates, str | bytes | bytearray):
        raise ValueError("candidates must be a sequence")
    candidate_items = tuple(candidates)
    if any(type(candidate) is not TowerCandidate for candidate in candidate_items):
        raise ValueError("candidates must contain TowerCandidate instances")
    if (
        corridor_bounds_absolute_m is not None
        and type(corridor_bounds_absolute_m) is not TowerCandidateBounds
    ):
        raise ValueError("corridor_bounds_absolute_m must be a TowerCandidateBounds or null")
    canonical = _canonical_candidate_order(candidate_items)
    payload = _candidate_set_digest_payload(
        schema_version=schema_version,
        project_id=project_id,
        source_binding=source_binding,
        coordinate_context=coordinate_context,
        detector_provenance=detector_provenance,
        candidates=canonical,
        corridor_bounds_absolute_m=corridor_bounds_absolute_m,
    )
    return frv_digest(payload)


@dataclass(frozen=True, slots=True)
class TowerCandidateSet:
    """Immutable ordered set of tower candidates for one detector generation."""

    schema_version: str
    candidate_set_id: str
    project_id: str
    generated_at: str
    source_binding: SourceIdentityBinding
    coordinate_context: CoordinateContext
    detector_provenance: DetectorProvenance
    candidates: tuple[TowerCandidate, ...]
    candidate_count: int
    corridor_bounds_absolute_m: TowerCandidateBounds | None = None
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_CANDIDATE_SET_SCHEMA_VERSION:
            raise ValueError("unsupported tower-candidate-set schema version")
        _opaque_identifier(self.candidate_set_id, "candidate_set_id")
        _project_id(self.project_id)
        _timestamp(self.generated_at, "generated_at")
        if type(self.source_binding) is not SourceIdentityBinding:
            raise ValueError("source_binding must be a SourceIdentityBinding")
        if type(self.coordinate_context) is not CoordinateContext:
            raise ValueError("coordinate_context must be a CoordinateContext")
        if self.coordinate_context != self.source_binding.coordinate_context:
            raise ValueError("coordinate_context must match source_binding")
        if type(self.detector_provenance) is not DetectorProvenance:
            raise ValueError("detector_provenance must be a DetectorProvenance")
        if not isinstance(self.candidates, Sequence) or isinstance(
            self.candidates, str | bytes | bytearray
        ):
            raise ValueError("candidates must be a sequence")
        candidates = tuple(self.candidates)
        if any(type(candidate) is not TowerCandidate for candidate in candidates):
            raise ValueError("candidates must contain TowerCandidate instances")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise ValueError("candidate_ids must be unique within a tower-candidate set")
        for candidate in candidates:
            if (
                candidate.coordinate_frame is not self.coordinate_context.frame
                or candidate.coordinate_unit is not self.coordinate_context.units
            ):
                raise ValueError(
                    "candidate coordinate frame/unit must match the set coordinate context"
                )
        if type(self.candidate_count) is not int or self.candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative integer")
        if self.candidate_count != len(candidates):
            raise ValueError("candidate_count must exactly match the number of candidate records")
        if (
            self.corridor_bounds_absolute_m is not None
            and type(self.corridor_bounds_absolute_m) is not TowerCandidateBounds
        ):
            raise ValueError("corridor_bounds_absolute_m must be a TowerCandidateBounds or null")
        # Canonicalize the stored candidate sequence so benign construction
        # order never changes the serialization or content digest.
        candidates = _canonical_candidate_order(candidates)
        object.__setattr__(self, "candidates", candidates)
        expected = derive_tower_candidate_set_digest(
            schema_version=self.schema_version,
            project_id=self.project_id,
            source_binding=self.source_binding,
            coordinate_context=self.coordinate_context,
            detector_provenance=self.detector_provenance,
            candidates=candidates,
            corridor_bounds_absolute_m=self.corridor_bounds_absolute_m,
        )
        if self.digest != expected:
            raise ValueError("digest does not match the tower-candidate-set content")

    @classmethod
    def create(
        cls,
        *,
        candidate_set_id: str,
        project_id: str,
        generated_at: str,
        source_binding: SourceIdentityBinding,
        coordinate_context: CoordinateContext,
        detector_provenance: DetectorProvenance,
        candidates: Sequence[TowerCandidate],
        corridor_bounds_absolute_m: TowerCandidateBounds | None = None,
    ) -> TowerCandidateSet:
        """Construct a set, deriving its content digest automatically."""
        digest = derive_tower_candidate_set_digest(
            schema_version=TOWER_CANDIDATE_SET_SCHEMA_VERSION,
            project_id=project_id,
            source_binding=source_binding,
            coordinate_context=coordinate_context,
            detector_provenance=detector_provenance,
            candidates=candidates,
            corridor_bounds_absolute_m=corridor_bounds_absolute_m,
        )
        return cls(
            schema_version=TOWER_CANDIDATE_SET_SCHEMA_VERSION,
            candidate_set_id=candidate_set_id,
            project_id=project_id,
            generated_at=generated_at,
            source_binding=source_binding,
            coordinate_context=coordinate_context,
            detector_provenance=detector_provenance,
            candidates=tuple(candidates),
            candidate_count=len(tuple(candidates)),
            corridor_bounds_absolute_m=corridor_bounds_absolute_m,
            digest=digest,
        )

    @property
    def reference(self) -> CandidateSetReference:
        return CandidateSetReference(
            candidate_set_id=self.candidate_set_id,
            candidate_set_digest=self.digest,
        )

    def to_digest_dict(self) -> dict[str, Any]:
        return _candidate_set_digest_payload(
            schema_version=self.schema_version,
            project_id=self.project_id,
            source_binding=self.source_binding,
            coordinate_context=self.coordinate_context,
            detector_provenance=self.detector_provenance,
            candidates=self.candidates,
            corridor_bounds_absolute_m=self.corridor_bounds_absolute_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_set_id": self.candidate_set_id,
            "project_id": self.project_id,
            "generated_at": self.generated_at,
            "source_binding": self.source_binding.to_dict(),
            "coordinate_context": self.coordinate_context.to_dict(),
            "detector_provenance": self.detector_provenance.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "candidate_count": self.candidate_count,
            "corridor_bounds_absolute_m": (
                self.corridor_bounds_absolute_m.to_dict()
                if self.corridor_bounds_absolute_m is not None
                else None
            ),
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateSet:
        values = _from_dict_values(cls, data)
        values["coordinate_context"] = _coordinate_context_from_dict(values["coordinate_context"])
        values["source_binding"] = _source_binding_from_dict(values["source_binding"])
        values["detector_provenance"] = DetectorProvenance.from_dict(values["detector_provenance"])
        values["candidates"] = tuple(
            TowerCandidate.from_dict(item)
            for item in _json_array(values["candidates"], "candidates")
        )
        if values["corridor_bounds_absolute_m"] is not None:
            values["corridor_bounds_absolute_m"] = TowerCandidateBounds.from_dict(
                values["corridor_bounds_absolute_m"]
            )
        return cls(**values)

    def to_json_bytes(self) -> bytes:
        return _strict_json_dumps(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> TowerCandidateSet:
        return cls.from_dict(_strict_json_loads(data))


# ---------------------------------------------------------------------------
# Review events and ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TowerCandidateReviewEvent:
    """One immutable, append-only semantic review event for one candidate."""

    schema_version: str
    candidate_set_reference: CandidateSetReference
    candidate_id: str
    review_revision: int
    decision: ReviewDecision
    occurred_at: str
    reason: str
    proposal_xyz_absolute_m: tuple[float, float, float]
    refinement_state: RefinementState
    proposal_bounds_absolute_m: TowerCandidateBounds | None = None
    refined_raw_geometry: TowerGeometry | None = None
    refinement_evidence: TowerCandidateRawRefinementEvidence | None = None
    operator_identity: str | None = None
    review_adjusted_seed_xyz_absolute_m: tuple[float, float, float] | None = None
    notes: str | None = None
    prior_event_revision: int | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported tower-candidate-review-event schema version")
        if type(self.candidate_set_reference) is not CandidateSetReference:
            raise ValueError("candidate_set_reference must be a CandidateSetReference")
        _candidate_id(self.candidate_id)
        _positive_int(self.review_revision, "review_revision")
        _enum(self.decision, ReviewDecision, "decision")
        _timestamp(self.occurred_at, "occurred_at")
        _exact_string(self.reason, "reason")
        object.__setattr__(
            self,
            "proposal_xyz_absolute_m",
            _xyz(self.proposal_xyz_absolute_m, "proposal_xyz_absolute_m"),
        )
        if (
            self.proposal_bounds_absolute_m is not None
            and type(self.proposal_bounds_absolute_m) is not TowerCandidateBounds
        ):
            raise ValueError("proposal_bounds_absolute_m must be a TowerCandidateBounds or null")
        _enum(self.refinement_state, RefinementState, "refinement_state")
        if (
            self.refined_raw_geometry is not None
            and type(self.refined_raw_geometry) is not TowerGeometry
        ):
            raise ValueError("refined_raw_geometry must be a TowerGeometry or null")
        if (
            self.refinement_evidence is not None
            and type(self.refinement_evidence) is not TowerCandidateRawRefinementEvidence
        ):
            raise ValueError(
                "refinement_evidence must be a TowerCandidateRawRefinementEvidence or null"
            )
        if self.operator_identity is not None:
            _exact_string(self.operator_identity, "operator_identity")
        if self.review_adjusted_seed_xyz_absolute_m is not None:
            object.__setattr__(
                self,
                "review_adjusted_seed_xyz_absolute_m",
                _xyz(
                    self.review_adjusted_seed_xyz_absolute_m,
                    "review_adjusted_seed_xyz_absolute_m",
                ),
            )
        if self.notes is not None:
            _exact_string(self.notes, "notes")
        if self.prior_event_revision is not None:
            _positive_int(self.prior_event_revision, "prior_event_revision")
            if self.prior_event_revision >= self.review_revision:
                raise ValueError("prior_event_revision must not point forward")
        if self.decision is ReviewDecision.ACCEPTED:
            if (
                self.refinement_state is not RefinementState.RAW_REFINED
                or self.refined_raw_geometry is None
                or self.refinement_evidence is None
            ):
                raise ValueError(
                    "ACCEPTED review requires RAW_REFINED geometry and "
                    "non-empty refinement evidence"
                )
        if self.refinement_state is RefinementState.RAW_REFINED:
            if self.refined_raw_geometry is None:
                raise ValueError("RAW_REFINED review state requires refined_raw_geometry")
            if self.refinement_evidence is None:
                raise ValueError("RAW_REFINED review state requires refinement evidence")
        if (
            self.refined_raw_geometry is not None
            and self.refinement_state is not RefinementState.RAW_REFINED
        ):
            raise ValueError("refined_raw_geometry requires RAW_REFINED review state")
        if (
            self.refinement_evidence is not None
            and self.refinement_state is not RefinementState.RAW_REFINED
        ):
            raise ValueError("refinement evidence requires RAW_REFINED review state")
        if self.decision is ReviewDecision.SUPERSEDED and self.prior_event_revision is None:
            raise ValueError("SUPERSEDED review decision requires prior_event_revision")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_set_reference": self.candidate_set_reference.to_dict(),
            "candidate_id": self.candidate_id,
            "review_revision": self.review_revision,
            "decision": self.decision.value,
            "occurred_at": self.occurred_at,
            "reason": self.reason,
            "proposal_xyz_absolute_m": list(self.proposal_xyz_absolute_m),
            "refinement_state": self.refinement_state.value,
            "proposal_bounds_absolute_m": (
                self.proposal_bounds_absolute_m.to_dict()
                if self.proposal_bounds_absolute_m is not None
                else None
            ),
            "refined_raw_geometry": (
                self.refined_raw_geometry.to_dict()
                if self.refined_raw_geometry is not None
                else None
            ),
            "refinement_evidence": (
                self.refinement_evidence.to_dict()
                if self.refinement_evidence is not None
                else None
            ),
            "operator_identity": self.operator_identity,
            "review_adjusted_seed_xyz_absolute_m": (
                list(self.review_adjusted_seed_xyz_absolute_m)
                if self.review_adjusted_seed_xyz_absolute_m is not None
                else None
            ),
            "notes": self.notes,
            "prior_event_revision": self.prior_event_revision,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateReviewEvent:
        values = _from_dict_values(cls, data)
        values["candidate_set_reference"] = CandidateSetReference.from_dict(
            values["candidate_set_reference"]
        )
        values["decision"] = _enum_from_value(values["decision"], ReviewDecision, "decision")
        values["proposal_xyz_absolute_m"] = _tuple_from_json(
            values["proposal_xyz_absolute_m"], "proposal_xyz_absolute_m"
        )
        if values["proposal_bounds_absolute_m"] is not None:
            values["proposal_bounds_absolute_m"] = TowerCandidateBounds.from_dict(
                values["proposal_bounds_absolute_m"]
            )
        values["refinement_state"] = _enum_from_value(
            values["refinement_state"], RefinementState, "refinement_state"
        )
        if values["refined_raw_geometry"] is not None:
            values["refined_raw_geometry"] = _tower_geometry_from_dict(
                values["refined_raw_geometry"]
            )
        if values["refinement_evidence"] is not None:
            values["refinement_evidence"] = TowerCandidateRawRefinementEvidence.from_dict(
                values["refinement_evidence"]
            )
        if values["review_adjusted_seed_xyz_absolute_m"] is not None:
            values["review_adjusted_seed_xyz_absolute_m"] = _tuple_from_json(
                values["review_adjusted_seed_xyz_absolute_m"],
                "review_adjusted_seed_xyz_absolute_m",
            )
        return cls(**values)

    def to_json_bytes(self) -> bytes:
        return _strict_json_dumps(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> TowerCandidateReviewEvent:
        return cls.from_dict(_strict_json_loads(data))


@dataclass(frozen=True, slots=True)
class TowerCandidateReviewLedger:
    """Immutable append-only ledger of review events for declared candidate sets."""

    schema_version: str
    review_ledger_id: str
    candidate_sets: tuple[CandidateSetReference, ...]
    events: tuple[TowerCandidateReviewEvent, ...]

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION:
            raise ValueError("unsupported tower-candidate-review-ledger schema version")
        _opaque_identifier(self.review_ledger_id, "review_ledger_id")
        if not isinstance(self.candidate_sets, Sequence) or isinstance(
            self.candidate_sets, str | bytes | bytearray
        ):
            raise ValueError("candidate_sets must be a sequence")
        candidate_sets = tuple(self.candidate_sets)
        if any(type(item) is not CandidateSetReference for item in candidate_sets):
            raise ValueError("candidate_sets must contain CandidateSetReference instances")
        if len({item.candidate_set_id for item in candidate_sets}) != len(candidate_sets):
            raise ValueError("candidate_sets must be unique by candidate_set_id")
        if not isinstance(self.events, Sequence) or isinstance(
            self.events, str | bytes | bytearray
        ):
            raise ValueError("events must be a sequence")
        events = tuple(self.events)
        if any(type(event) is not TowerCandidateReviewEvent for event in events):
            raise ValueError("events must contain TowerCandidateReviewEvent instances")
        declared = {(item.candidate_set_id, item.candidate_set_digest) for item in candidate_sets}
        for event in events:
            key = (
                event.candidate_set_reference.candidate_set_id,
                event.candidate_set_reference.candidate_set_digest,
            )
            if key not in declared:
                raise ValueError(
                    "review event candidate-set reference must resolve within the ledger context"
                )
        events = sorted(
            events,
            key=lambda event: (
                event.candidate_set_reference.candidate_set_id,
                event.candidate_id,
                event.review_revision,
            ),
        )
        candidate_sets = sorted(candidate_sets, key=lambda item: item.candidate_set_id)
        seen_revisions: dict[tuple[str, str], int] = {}
        for event in events:
            key = (event.candidate_set_reference.candidate_set_id, event.candidate_id)
            previous = seen_revisions.get(key)
            if previous is None:
                expected_revision = 1
                expected_prior = None
            else:
                expected_revision = previous + 1
                expected_prior = previous
            if event.review_revision != expected_revision:
                raise ValueError(
                    "review revisions for a candidate must be consecutive starting from 1"
                )
            if event.prior_event_revision != expected_prior:
                raise ValueError(
                    "prior_event_revision must link to the immediately preceding review revision"
                )
            seen_revisions[key] = event.review_revision
        object.__setattr__(self, "candidate_sets", candidate_sets)
        object.__setattr__(self, "events", events)

    def next_review_revision(
        self,
        candidate_set_reference: CandidateSetReference,
        candidate_id: str,
    ) -> int:
        """Return the next append-only review revision for one candidate (1 if none yet)."""
        if type(candidate_set_reference) is not CandidateSetReference:
            raise ValueError("candidate_set_reference must be a CandidateSetReference")
        _candidate_id(candidate_id)
        current = self._revision_chain(candidate_set_reference, candidate_id)
        return (current[-1] if current else 0) + 1

    def effective_decision(
        self,
        candidate_set_reference: CandidateSetReference,
        candidate_id: str,
    ) -> ReviewDecision | None:
        """Project the effective decision: the decision of the newest review event."""
        if type(candidate_set_reference) is not CandidateSetReference:
            raise ValueError("candidate_set_reference must be a CandidateSetReference")
        _candidate_id(candidate_id)
        chain = self._review_chain(candidate_set_reference, candidate_id)
        if not chain:
            return None
        return chain[-1].decision

    def _review_chain(
        self,
        candidate_set_reference: CandidateSetReference,
        candidate_id: str,
    ) -> tuple[TowerCandidateReviewEvent, ...]:
        return tuple(
            event
            for event in self.events
            if event.candidate_set_reference == candidate_set_reference
            and event.candidate_id == candidate_id
        )

    def _revision_chain(
        self,
        candidate_set_reference: CandidateSetReference,
        candidate_id: str,
    ) -> tuple[int, ...]:
        return tuple(
            event.review_revision
            for event in self._review_chain(candidate_set_reference, candidate_id)
        )

    def with_event(self, event: TowerCandidateReviewEvent) -> TowerCandidateReviewLedger:
        """Return a new ledger with one additional immutable review event appended."""
        if type(event) is not TowerCandidateReviewEvent:
            raise ValueError("event must be a TowerCandidateReviewEvent")
        return TowerCandidateReviewLedger(
            schema_version=self.schema_version,
            review_ledger_id=self.review_ledger_id,
            candidate_sets=self.candidate_sets,
            events=(*self.events, event),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "review_ledger_id": self.review_ledger_id,
            "candidate_sets": [item.to_dict() for item in self.candidate_sets],
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateReviewLedger:
        values = _from_dict_values(cls, data)
        values["candidate_sets"] = tuple(
            CandidateSetReference.from_dict(item)
            for item in _json_array(values["candidate_sets"], "candidate_sets")
        )
        values["events"] = tuple(
            TowerCandidateReviewEvent.from_dict(item)
            for item in _json_array(values["events"], "events")
        )
        return cls(**values)

    def to_json_bytes(self) -> bytes:
        return _strict_json_dumps(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> TowerCandidateReviewLedger:
        return cls.from_dict(_strict_json_loads(data))


# ---------------------------------------------------------------------------
# Candidate confirmation (T### allocation happens in V5A1-B, never here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TowerCandidateConfirmation:
    """Immutable mapping from an accepted candidate review to a confirmed tower.

    ``TowerCandidateConfirmation`` records the *future* mapping
    ``candidate-set reference + P### + accepted review revision -> T### +
    tower revision``.  This contract never allocates a ``T###``; it only
    validates that a supplied ``T###`` is a canonical existing tower ID.
    """

    schema_version: str
    candidate_set_reference: CandidateSetReference
    candidate_id: str
    accepted_review_revision: int
    confirmed_tower: TowerRevisionReference
    confirmed_at: str
    detector_provenance: DetectorProvenance | None = None
    reviewer_identity: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION:
            raise ValueError("unsupported tower-candidate-confirmation schema version")
        if type(self.candidate_set_reference) is not CandidateSetReference:
            raise ValueError("candidate_set_reference must be a CandidateSetReference")
        _candidate_id(self.candidate_id)
        _positive_int(self.accepted_review_revision, "accepted_review_revision")
        if type(self.confirmed_tower) is not TowerRevisionReference:
            raise ValueError("confirmed_tower must be a TowerRevisionReference")
        if (
            self.detector_provenance is not None
            and type(self.detector_provenance) is not DetectorProvenance
        ):
            raise ValueError("detector_provenance must be a DetectorProvenance or null")
        if self.reviewer_identity is not None:
            _exact_string(self.reviewer_identity, "reviewer_identity")
        _timestamp(self.confirmed_at, "confirmed_at")
        if self.notes is not None:
            _exact_string(self.notes, "notes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_set_reference": self.candidate_set_reference.to_dict(),
            "candidate_id": self.candidate_id,
            "accepted_review_revision": self.accepted_review_revision,
            "confirmed_tower": self.confirmed_tower.to_dict(),
            "confirmed_at": self.confirmed_at,
            "detector_provenance": (
                self.detector_provenance.to_dict()
                if self.detector_provenance is not None
                else None
            ),
            "reviewer_identity": self.reviewer_identity,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TowerCandidateConfirmation:
        values = _from_dict_values(cls, data)
        values["candidate_set_reference"] = CandidateSetReference.from_dict(
            values["candidate_set_reference"]
        )
        values["confirmed_tower"] = _tower_revision_from_dict(values["confirmed_tower"])
        if values["detector_provenance"] is not None:
            values["detector_provenance"] = DetectorProvenance.from_dict(
                values["detector_provenance"]
            )
        return cls(**values)

    def to_json_bytes(self) -> bytes:
        return _strict_json_dumps(self.to_dict())

    @classmethod
    def from_json_bytes(cls, data: bytes) -> TowerCandidateConfirmation:
        return cls.from_dict(_strict_json_loads(data))


def _candidate_by_id(
    candidate_set: TowerCandidateSet,
    candidate_id: str,
) -> TowerCandidate | None:
    for candidate in candidate_set.candidates:
        if candidate.candidate_id == candidate_id:
            return candidate
    return None


def validate_tower_candidate_review_ledger(
    candidate_set: TowerCandidateSet,
    ledger: TowerCandidateReviewLedger,
) -> None:
    """Ground an immutable review ledger against the actual immutable candidate set.

    Every review event must resolve exactly to the actual set's
    ``(candidate_set_id, candidate_set_digest)`` reference, reference a
    candidate that actually exists in that set, carry an audit proposal that
    matches the immutable candidate proposal, and (when refined) bind its
    refinement evidence to the set's authoritative source identity.
    """

    if type(candidate_set) is not TowerCandidateSet:
        raise ValueError("candidate_set must be a TowerCandidateSet")
    if type(ledger) is not TowerCandidateReviewLedger:
        raise ValueError("ledger must be a TowerCandidateReviewLedger")
    if tuple(ledger.candidate_sets) != (candidate_set.reference,):
        raise ValueError(
            "ledger must declare exactly the candidate-set reference of the actual set"
        )
    actual_reference = candidate_set.reference
    for event in ledger.events:
        if event.candidate_set_reference != actual_reference:
            raise ValueError(
                "review event candidate-set reference must match the actual candidate set"
            )
        candidate = _candidate_by_id(candidate_set, event.candidate_id)
        if candidate is None:
            raise ValueError(
                f"review event references candidate {event.candidate_id} "
                "absent from the candidate set"
            )
        if event.proposal_xyz_absolute_m != candidate.candidate_xyz_absolute_m:
            raise ValueError(
                "review event proposal must match the immutable candidate proposal"
            )
        if event.proposal_bounds_absolute_m != candidate.candidate_bounds_absolute_m:
            raise ValueError(
                "review event proposal bounds must match the immutable candidate proposal"
            )
        evidence = event.refinement_evidence
        if evidence is not None and (
            evidence.source_content_identity_digest
            != candidate_set.source_binding.source_content_identity_digest
            or evidence.source_identity_record_digest
            != candidate_set.source_binding.source_identity_record_digest
        ):
            raise ValueError(
                "refinement evidence source linkage must match the candidate set source"
            )


def validate_tower_candidate_confirmation(
    candidate_set: TowerCandidateSet,
    ledger: TowerCandidateReviewLedger,
    confirmation: TowerCandidateConfirmation,
) -> None:
    """Semantically validate a confirmation against the actual set and actual ledger.

    Structural JSON validity is not semantic confirmation validity.  A
    confirmation is only semantically valid when it resolves through the
    actual immutable ``TowerCandidateSet`` and the actual immutable review
    ledger: the set reference must match exactly, the candidate must exist in
    that set, the referenced review revision must be the unique effective
    ACCEPTED decision carrying RAW_REFINED geometry and evidence, and no later
    review may have superseded it.  Rejected and superseded observations can
    never become confirmations.
    """

    if type(confirmation) is not TowerCandidateConfirmation:
        raise ValueError("confirmation must be a TowerCandidateConfirmation")
    if type(ledger) is not TowerCandidateReviewLedger:
        raise ValueError("ledger must be a TowerCandidateReviewLedger")
    if type(candidate_set) is not TowerCandidateSet:
        raise ValueError("candidate_set must be a TowerCandidateSet")
    validate_tower_candidate_review_ledger(candidate_set, ledger)
    if confirmation.candidate_set_reference != candidate_set.reference:
        raise ValueError(
            "confirmation candidate-set reference must resolve exactly to the actual candidate set"
        )
    if _candidate_by_id(candidate_set, confirmation.candidate_id) is None:
        raise ValueError("confirmation candidate must exist in the actual candidate set")
    candidate_events = [
        event
        for event in ledger.events
        if event.candidate_id == confirmation.candidate_id
    ]
    source_events = [
        event
        for event in candidate_events
        if event.review_revision == confirmation.accepted_review_revision
    ]
    if len(source_events) != 1:
        raise ValueError("confirmation must reference exactly one review event for the candidate")
    source = source_events[0]
    if source.decision is not ReviewDecision.ACCEPTED:
        raise ValueError("confirmation source review must be semantically ACCEPTED")
    if (
        source.refinement_state is not RefinementState.RAW_REFINED
        or source.refined_raw_geometry is None
        or source.refinement_evidence is None
    ):
        raise ValueError(
            "confirmation source review must carry RAW_REFINED geometry and refinement evidence"
        )
    if any(event.review_revision > source.review_revision for event in candidate_events):
        raise ValueError("confirmation source review must remain the effective decision")


__all__ = [
    "TOWER_CANDIDATE_SET_SCHEMA_VERSION",
    "TOWER_CANDIDATE_REVIEW_EVENT_SCHEMA_VERSION",
    "TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION",
    "TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION",
    "TOWER_CANDIDATE_SET_DIGEST_DOMAIN",
    "CandidateOrigin",
    "DetectorKind",
    "RefinementState",
    "ReviewDecision",
    "TowerCandidateQualityFlag",
    "DetectorProvenance",
    "TowerCandidateManualAddProvenance",
    "TowerCandidateBounds",
    "TowerCandidateRawRefinementEvidence",
    "TowerCandidate",
    "CandidateSetReference",
    "TowerCandidateSet",
    "TowerCandidateReviewEvent",
    "TowerCandidateReviewLedger",
    "TowerCandidateConfirmation",
    "derive_tower_candidate_set_digest",
    "validate_tower_candidate_review_ledger",
    "validate_tower_candidate_confirmation",
]
