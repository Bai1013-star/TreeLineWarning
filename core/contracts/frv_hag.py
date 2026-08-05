"""FRV-A2 P-HAG provider contracts (PLAN-FROZEN).

P-HAG consumes a finalized P-SOURCE result and emits full-resolution HAG
evidence over the complete FULL_SOURCE domain.  This module freezes the
provider config, runtime chunk, success proof, finalized result, load error,
session protocol, validators, digest framings, and the FRVA2BND1 adapter.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Literal, Protocol

import numpy as np

from core.contracts.frv import (
    FrvErrorCode,
    FrvHagEvidenceState,
    FrvQualityFlag,
    FrvReasonFlag,
    SourceIdentity,
    frv_digest,
    validate_frv_flag_mask,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2DomainKind,
    FrvA2EvidenceClass,
    FrvA2FailureRecord,
    FrvA2ProviderBindingEvidence,
    FrvA2ProviderReadiness,
    FrvA2ProviderRole,
    _derive_artifact_reference_id,
    frv_a2_provider_binding_digest,
    validate_frv_a2_failure_record,
)
from core.contracts.frv_source import (
    FrvSourcePassThroughDimension,
)

# ---------------------------------------------------------------------------
# Public constants (frozen names and values)
# ---------------------------------------------------------------------------

FRV_HAG_PROVIDER_KEY = "hag_provider"
FRV_HAG_PROVIDER_INTERFACE_VERSION = "frv-hag-provider-v1"
FRV_HAG_PROVIDER_SCHEMA_VERSION = "frv-hag-provider-result-v1"
FRV_HAG_CONFIG_SCHEMA_VERSION = "frv-hag-provider-config-v1"
FRV_HAG_PROOF_SCHEMA_VERSION = "frv-hag-provider-proof-v1"
FRV_HAG_ALGORITHM_VERSION = "frv-hag-sparse-grid-idw2-v1"
FRV_HAG_CONFIG_DIGEST_DOMAIN = "FRVHAGCFG1"
FRV_HAG_SOURCE_PROOF_DIGEST_DOMAIN = "FRVHAGSRC1"
FRV_HAG_GRID_DIGEST_DOMAIN = "FRVHAGGRID1"
FRV_HAG_POINT_CONTENT_DIGEST_DOMAIN = "FRVHAGPCD1"
FRV_HAG_PROOF_DIGEST_DOMAIN = "FRVHAGPRF1"

_FRV_HAG_ARTIFACT_REFERENCE_PREFIX = "frv-a2-ref-v1|"

# ---------------------------------------------------------------------------
# Private strict validation helpers (contract layer only)
# ---------------------------------------------------------------------------


def _require_exact_str(value: Any, name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in str")
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _require_sha(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a positive built-in int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _require_enum(value: Any, enum_type: type, name: str) -> Any:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _require_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an immutable tuple")
    return value


def _exact_fields(data: Any, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} must be a mapping")
    actual = set(data)
    missing = fields - actual
    unknown = actual - fields
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{name} has unknown fields: {sorted(unknown)}")
    return dict(data)


def _tuple_from_json(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _enum_from_value(value: Any, enum_type: type, name: str) -> Any:
    text = _require_exact_str(value, name)
    try:
        return enum_type(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


def _validate_vec3(value: Any, name: str) -> tuple[float, float, float]:
    items = _require_tuple(value, name)
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (
        _require_finite_float(items[0], f"{name}[0]"),
        _require_finite_float(items[1], f"{name}[1]"),
        _require_finite_float(items[2], f"{name}[2]"),
    )


def _validate_enum_tuple(value: Any, enum_type: type, name: str) -> tuple[Any, ...]:
    items = _require_tuple(value, name)
    order = {member: index for index, member in enumerate(enum_type)}
    seen: set[Any] = set()
    validated = []
    for item in items:
        _require_enum(item, enum_type, f"{name}[]")
        if item in seen:
            raise ValueError(f"{name} must be unique")
        seen.add(item)
        validated.append(item)
    if tuple(sorted(validated, key=order.__getitem__)) != tuple(validated):
        raise ValueError(f"{name} must use declaration order")
    return tuple(validated)


def _require_artifact_reference_id(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if not text.startswith(_FRV_HAG_ARTIFACT_REFERENCE_PREFIX):
        raise ValueError(f"{name} must start with {_FRV_HAG_ARTIFACT_REFERENCE_PREFIX!r}")
    return text


def _normalize_zero(value: float) -> float:
    """Return +0.0 for any exact-zero float, otherwise the finite value."""
    return 0.0 if value == 0.0 else float(value)


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _require_float_array(
    value: Any, name: str, *, size: int, zero_normalized: bool
) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != np.dtype("<f8"):
        raise ValueError(f"{name} must use dtype <f8")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if zero_normalized:
        for item in value:
            if _is_negative_zero(float(item)):
                raise ValueError(f"{name} must not contain IEEE negative zero")


def _require_int_array(value: Any, name: str, *, dtype: np.dtype, size: int) -> None:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a numpy ndarray")
    if value.dtype != dtype:
        raise ValueError(f"{name} must use dtype {dtype.str}")
    if value.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not value.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous")
    if value.flags.writeable:
        raise ValueError(f"{name} must be read-only")


def _stream_index_domain_digest(total: int, source_content_identity_digest: str) -> str:
    """Return the FRVA2IDX1 digest over the global 0..N-1 domain, in-stream.

    Byte-identical to ``frv_a2_index_domain_digest`` for ``tuple(range(total))``
    but hashes the lazy ``range`` and never builds an N-length collection.
    """

    hasher = hashlib.sha256()
    hasher.update(b"FRVA2IDX1\0")
    hasher.update(b"ORIGINAL_INDEX")
    hasher.update(b"\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(FrvA2DomainKind.FULL_SOURCE.value.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", total))
    for index in range(total):
        hasher.update(struct.pack("<Q", index))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# FRVHAGCFG1 provider-config digest
# ---------------------------------------------------------------------------


def derive_frv_hag_provider_config_digest(
    *,
    schema_version: str,
    provider_key: str,
    provider_role: FrvA2ProviderRole,
    interface_version: str,
    algorithm_version: str,
    cell_size_m: float,
    support_radius_m: float,
    min_support_cells: int,
    max_support_cells: int,
    negative_hag_tolerance_m: float,
    classification_dimension: str,
) -> str:
    """Return the FRVHAGCFG1 digest for the frozen provider-config payload."""

    payload = {
        "digest_domain": FRV_HAG_CONFIG_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "provider_key": provider_key,
        "provider_role": provider_role.value,
        "interface_version": interface_version,
        "algorithm_version": algorithm_version,
        "cell_size_m": cell_size_m,
        "support_radius_m": support_radius_m,
        "min_support_cells": min_support_cells,
        "max_support_cells": max_support_cells,
        "negative_hag_tolerance_m": negative_hag_tolerance_m,
        "classification_dimension": classification_dimension,
    }
    return frv_digest(payload)


# ---------------------------------------------------------------------------
# FrvHagProviderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvHagProviderConfig:
    """Exact-schema P-HAG provider configuration (digest domain FRVHAGCFG1)."""

    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    cell_size_m: float
    support_radius_m: float
    min_support_cells: int
    max_support_cells: int
    negative_hag_tolerance_m: float
    classification_dimension: str
    provider_config_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_HAG_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported P-HAG provider config schema version")
        _require_exact_str(self.provider_key, "provider_key")
        if self.provider_key != FRV_HAG_PROVIDER_KEY:
            raise ValueError("provider_key must be hag_provider")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        if self.provider_role is not FrvA2ProviderRole.HAG:
            raise ValueError("provider_role must be HAG")
        _require_exact_str(self.interface_version, "interface_version")
        if self.interface_version != FRV_HAG_PROVIDER_INTERFACE_VERSION:
            raise ValueError("interface_version must match the frozen interface version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != FRV_HAG_ALGORITHM_VERSION:
            raise ValueError("algorithm_version must match the frozen algorithm version")
        object.__setattr__(
            self, "cell_size_m", _require_finite_float(self.cell_size_m, "cell_size_m")
        )
        object.__setattr__(
            self,
            "support_radius_m",
            _require_finite_float(self.support_radius_m, "support_radius_m"),
        )
        object.__setattr__(
            self,
            "min_support_cells",
            _require_positive_int(self.min_support_cells, "min_support_cells"),
        )
        object.__setattr__(
            self,
            "max_support_cells",
            _require_positive_int(self.max_support_cells, "max_support_cells"),
        )
        object.__setattr__(
            self,
            "negative_hag_tolerance_m",
            _require_finite_float(self.negative_hag_tolerance_m, "negative_hag_tolerance_m"),
        )
        _require_exact_str(self.classification_dimension, "classification_dimension")
        if self.classification_dimension != "classification":
            raise ValueError("classification_dimension must be classification")
        if self.support_radius_m < self.cell_size_m:
            raise ValueError("support_radius_m must be >= cell_size_m")
        if self.max_support_cells < self.min_support_cells:
            raise ValueError("max_support_cells must be >= min_support_cells")
        _require_sha(self.provider_config_digest, "provider_config_digest")
        computed = derive_frv_hag_provider_config_digest(
            schema_version=self.schema_version,
            provider_key=self.provider_key,
            provider_role=self.provider_role,
            interface_version=self.interface_version,
            algorithm_version=self.algorithm_version,
            cell_size_m=self.cell_size_m,
            support_radius_m=self.support_radius_m,
            min_support_cells=self.min_support_cells,
            max_support_cells=self.max_support_cells,
            negative_hag_tolerance_m=self.negative_hag_tolerance_m,
            classification_dimension=self.classification_dimension,
        )
        if self.provider_config_digest != computed:
            raise ValueError("provider_config_digest does not match canonical FRVHAGCFG1 fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_role": self.provider_role.value,
            "interface_version": self.interface_version,
            "algorithm_version": self.algorithm_version,
            "cell_size_m": self.cell_size_m,
            "support_radius_m": self.support_radius_m,
            "min_support_cells": self.min_support_cells,
            "max_support_cells": self.max_support_cells,
            "negative_hag_tolerance_m": self.negative_hag_tolerance_m,
            "classification_dimension": self.classification_dimension,
            "provider_config_digest": self.provider_config_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvHagProviderConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_role"] = _enum_from_value(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        return cls(**values)


def validate_frv_hag_provider_config(config: FrvHagProviderConfig) -> None:
    if type(config) is not FrvHagProviderConfig:
        raise ValueError("config must be a FrvHagProviderConfig")
    computed = derive_frv_hag_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        cell_size_m=config.cell_size_m,
        support_radius_m=config.support_radius_m,
        min_support_cells=config.min_support_cells,
        max_support_cells=config.max_support_cells,
        negative_hag_tolerance_m=config.negative_hag_tolerance_m,
        classification_dimension=config.classification_dimension,
    )
    if config.provider_config_digest != computed:
        raise ValueError("provider_config_digest does not match canonical FRVHAGCFG1 fields")


# ---------------------------------------------------------------------------
# FrvHagChunk (runtime-only immutable contract)
# ---------------------------------------------------------------------------

_VALID_HAG_STATES = frozenset(FrvHagEvidenceState)


@dataclass(frozen=True, slots=True)
class FrvHagChunk:
    """Read-only P-HAG chunk with a one-to-one source-chunk mapping."""

    source_id: str
    chunk_id: int
    start_index: int
    point_count: int
    original_index: np.ndarray
    ground_support: np.ndarray
    support_cell_count: np.ndarray
    ground_z_m: np.ndarray
    raw_hag_m: np.ndarray
    clamped_hag_m: np.ndarray
    hag_evidence_state: np.ndarray
    reason_flags: np.ndarray
    quality_flags: np.ndarray
    is_last: bool

    def __post_init__(self) -> None:
        _require_exact_str(self.source_id, "source_id")
        object.__setattr__(
            self, "chunk_id", _require_nonnegative_int(self.chunk_id, "chunk_id")
        )
        object.__setattr__(
            self, "start_index", _require_nonnegative_int(self.start_index, "start_index")
        )
        point_count = _require_positive_int(self.point_count, "point_count")
        object.__setattr__(self, "point_count", point_count)
        if type(self.is_last) is not bool:
            raise ValueError("is_last must be a built-in bool")
        _require_int_array(
            self.original_index, "original_index", dtype=np.dtype("<i8"), size=point_count
        )
        _require_int_array(
            self.ground_support, "ground_support", dtype=np.dtype("bool"), size=point_count
        )
        _require_int_array(
            self.support_cell_count,
            "support_cell_count",
            dtype=np.dtype("<u2"),
            size=point_count,
        )
        _require_float_array(self.ground_z_m, "ground_z_m", size=point_count, zero_normalized=True)
        _require_float_array(self.raw_hag_m, "raw_hag_m", size=point_count, zero_normalized=True)
        _require_float_array(
            self.clamped_hag_m, "clamped_hag_m", size=point_count, zero_normalized=True
        )
        _require_int_array(
            self.hag_evidence_state, "hag_evidence_state", dtype=np.dtype("|u1"), size=point_count
        )
        _require_int_array(
            self.reason_flags, "reason_flags", dtype=np.dtype("<u4"), size=point_count
        )
        _require_int_array(
            self.quality_flags, "quality_flags", dtype=np.dtype("<u4"), size=point_count
        )


def validate_frv_hag_chunk(
    chunk: FrvHagChunk,
    *,
    provider_config: FrvHagProviderConfig,
) -> None:
    if type(chunk) is not FrvHagChunk:
        raise ValueError("chunk must be a FrvHagChunk")
    if type(provider_config) is not FrvHagProviderConfig:
        raise ValueError("provider_config must be a FrvHagProviderConfig")
    tolerance = provider_config.negative_hag_tolerance_m
    max_support = provider_config.max_support_cells
    missing_reason = int(
        FrvReasonFlag.INSUFFICIENT_EVIDENCE | FrvReasonFlag.HAG_MISSING
    )
    previous: int | None = None
    for row in range(chunk.point_count):
        index = int(chunk.original_index[row])
        if index < 0:
            raise ValueError("original_index must be nonnegative")
        if previous is not None and index <= previous:
            raise ValueError("original_index must be strictly increasing within chunk")
        previous = index
        state = int(chunk.hag_evidence_state[row])
        if state not in {int(member) for member in _VALID_HAG_STATES}:
            raise ValueError("hag_evidence_state must be a FrvHagEvidenceState value")
        supported = bool(chunk.ground_support[row])
        cell_count = int(chunk.support_cell_count[row])
        raw_hag = float(chunk.raw_hag_m[row])
        clamped = float(chunk.clamped_hag_m[row])
        reason = int(chunk.reason_flags[row])
        quality = int(chunk.quality_flags[row])
        if cell_count > max_support:
            raise ValueError("support_cell_count must not exceed max_support_cells")
        if state == int(FrvHagEvidenceState.HAG_MISSING):
            if supported or cell_count != 0:
                raise ValueError("HAG_MISSING rows must be unsupported with zero cell count")
            if (
                float(chunk.ground_z_m[row]) != 0.0
                or raw_hag != 0.0
                or clamped != 0.0
            ):
                raise ValueError("HAG_MISSING rows must carry positive-zero numeric values")
            if reason != missing_reason:
                raise ValueError(
                    "HAG_MISSING rows must carry INSUFFICIENT_EVIDENCE | HAG_MISSING"
                )
            if quality != 0:
                raise ValueError("HAG_MISSING rows must have NONE quality")
        elif state == int(FrvHagEvidenceState.HAG_OUT_OF_DOMAIN):
            if reason != int(FrvReasonFlag.HAG_OUT_OF_DOMAIN):
                raise ValueError("HAG_OUT_OF_DOMAIN rows must carry HAG_OUT_OF_DOMAIN reason")
            if quality != int(FrvQualityFlag.PROVIDER_INVALID):
                raise ValueError("HAG_OUT_OF_DOMAIN rows must carry PROVIDER_INVALID quality")
        else:
            if not supported or cell_count < 1:
                raise ValueError("supported HAG rows must be supported with nonzero cell count")
            if state == int(FrvHagEvidenceState.HAG_VALID_POSITIVE):
                if not (raw_hag > 0.0) or clamped != raw_hag:
                    raise ValueError("HAG_VALID_POSITIVE requires raw > 0 and clamped == raw")
                if reason != 0 or quality != 0:
                    raise ValueError("HAG_VALID_POSITIVE requires NONE reason and quality")
            elif state == int(FrvHagEvidenceState.HAG_VALID_NEAR_ZERO):
                if raw_hag != 0.0 or clamped != 0.0:
                    raise ValueError("HAG_VALID_NEAR_ZERO requires raw and clamped == +0.0")
                if reason != 0 or quality != 0:
                    raise ValueError("HAG_VALID_NEAR_ZERO requires NONE reason and quality")
            elif state == int(FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE):
                if not (-tolerance <= raw_hag < 0.0) or clamped != 0.0:
                    raise ValueError(
                        "HAG_WITHIN_NEGATIVE_TOLERANCE requires -tolerance <= raw < 0"
                    )
                if reason != 0 or quality != 0:
                    raise ValueError(
                        "HAG_WITHIN_NEGATIVE_TOLERANCE requires NONE reason and quality"
                    )
            elif state == int(FrvHagEvidenceState.HAG_INVALID):
                if not (raw_hag < -tolerance) or clamped != 0.0:
                    raise ValueError("HAG_INVALID requires raw < -tolerance and clamped == +0.0")
                if reason != int(FrvReasonFlag.HAG_INVALID):
                    raise ValueError("HAG_INVALID rows must carry HAG_INVALID reason")
                if quality != int(FrvQualityFlag.PROVIDER_INVALID):
                    raise ValueError("HAG_INVALID rows must carry PROVIDER_INVALID quality")
            else:
                raise ValueError("hag_evidence_state is not a recognized supported state")


# ---------------------------------------------------------------------------
# FRVHAGGRID1 ground-cell inventory digest
# ---------------------------------------------------------------------------


def frv_hag_ground_cell_inventory_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    provider_config_digest: str,
    ground_point_count: int,
    cells: tuple[tuple[int, int, float, int], ...],
) -> str:
    """Return the FRVHAGGRID1 digest over sorted unique ground cells."""

    sorted_cells = sorted(cells, key=lambda cell: (cell[0], cell[1]))
    hasher = hashlib.sha256()
    hasher.update(b"FRVHAGGRID1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(provider_config_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", ground_point_count))
    hasher.update(struct.pack("<Q", len(sorted_cells)))
    for ix, iy, min_z, count in sorted_cells:
        hasher.update(struct.pack("<q", ix))
        hasher.update(struct.pack("<q", iy))
        hasher.update(struct.pack("<d", _normalize_zero(min_z)))
        hasher.update(struct.pack("<Q", count))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# FRVHAGPCD1 HAG point-content digest
# ---------------------------------------------------------------------------


def frv_hag_point_content_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    provider_config_digest: str,
    source_point_count: int,
    chunks: Iterable[FrvHagChunk],
) -> str:
    """Return the FRVHAGPCD1 digest over the complete 0..N-1 HAG domain."""

    hasher = hashlib.sha256()
    hasher.update(b"FRVHAGPCD1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(provider_config_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", source_point_count))
    expected_index = 0
    for chunk in chunks:
        if type(chunk) is not FrvHagChunk:
            raise TypeError("chunks must contain FrvHagChunk values")
        for row in range(chunk.point_count):
            index = int(chunk.original_index[row])
            if index != expected_index:
                raise ValueError("chunks must cover the complete 0..N-1 domain")
            expected_index += 1
            hasher.update(struct.pack("<Q", index))
            hasher.update(struct.pack("<B", int(chunk.ground_support[row])))
            hasher.update(struct.pack("<H", int(chunk.support_cell_count[row])))
            hasher.update(
                struct.pack("<d", _normalize_zero(float(chunk.ground_z_m[row])))
            )
            hasher.update(struct.pack("<d", _normalize_zero(float(chunk.raw_hag_m[row]))))
            hasher.update(
                struct.pack("<d", _normalize_zero(float(chunk.clamped_hag_m[row])))
            )
            hasher.update(struct.pack("<B", int(chunk.hag_evidence_state[row])))
            hasher.update(struct.pack("<I", int(chunk.reason_flags[row])))
            hasher.update(struct.pack("<I", int(chunk.quality_flags[row])))
    if expected_index != source_point_count:
        raise ValueError("chunks must cover the complete 0..N-1 domain")
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# FrvHagProviderProof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvHagProviderProof:
    """Frozen, serializable exact-schema P-HAG success proof."""

    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    configuration_digest: str
    provider_config_digest: str
    source_artifact_reference_id: str
    artifact_reference_id: str
    artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    source_content_identity_digest: str
    source_identity_record_digest: str
    source_proof_digest: str
    source_binding_digest: str
    source_point_count: int
    domain_kind: FrvA2DomainKind
    domain_count: int
    original_index_domain_digest: str
    hag_point_content_digest: str
    ground_cell_inventory_digest: str
    ground_point_count: int
    supported_point_count: int
    unsupported_point_count: int
    negative_tolerance_exceeded_point_count: int
    hag_valid_positive_count: int
    hag_valid_near_zero_count: int
    hag_within_negative_tolerance_count: int
    hag_missing_count: int
    hag_invalid_count: int
    hag_out_of_domain_count: int
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...]
    quality_flags: int
    reason_flags: int
    error_codes: tuple[FrvErrorCode, ...]
    proof_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_HAG_PROOF_SCHEMA_VERSION:
            raise ValueError("unsupported P-HAG proof schema version")
        _require_exact_str(self.provider_key, "provider_key")
        if self.provider_key != FRV_HAG_PROVIDER_KEY:
            raise ValueError("provider_key must be hag_provider")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        if self.provider_role is not FrvA2ProviderRole.HAG:
            raise ValueError("provider_role must be HAG")
        _require_exact_str(self.interface_version, "interface_version")
        if self.interface_version != FRV_HAG_PROVIDER_INTERFACE_VERSION:
            raise ValueError("interface_version must match the frozen interface version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != FRV_HAG_ALGORITHM_VERSION:
            raise ValueError("algorithm_version must match the frozen algorithm version")
        _require_sha(self.configuration_digest, "configuration_digest")
        _require_sha(self.provider_config_digest, "provider_config_digest")
        _require_artifact_reference_id(
            self.source_artifact_reference_id, "source_artifact_reference_id"
        )
        _require_artifact_reference_id(self.artifact_reference_id, "artifact_reference_id")
        components = _require_tuple(self.artifact_components, "artifact_components")
        for index, component in enumerate(components):
            if type(component) is not FrvA2ArtifactComponentBinding:
                raise ValueError(
                    f"artifact_components[{index}] must be a FrvA2ArtifactComponentBinding"
                )
        _require_sha(self.source_content_identity_digest, "source_content_identity_digest")
        _require_sha(self.source_identity_record_digest, "source_identity_record_digest")
        _require_sha(self.source_proof_digest, "source_proof_digest")
        _require_sha(self.source_binding_digest, "source_binding_digest")
        object.__setattr__(
            self,
            "source_point_count",
            _require_positive_int(self.source_point_count, "source_point_count"),
        )
        _require_enum(self.domain_kind, FrvA2DomainKind, "domain_kind")
        if self.domain_kind is not FrvA2DomainKind.FULL_SOURCE:
            raise ValueError("domain_kind must be FULL_SOURCE")
        domain_count = _require_positive_int(self.domain_count, "domain_count")
        object.__setattr__(self, "domain_count", domain_count)
        if domain_count != self.source_point_count:
            raise ValueError("domain_count must equal source_point_count")
        _require_sha(self.original_index_domain_digest, "original_index_domain_digest")
        _require_sha(self.hag_point_content_digest, "hag_point_content_digest")
        _require_sha(self.ground_cell_inventory_digest, "ground_cell_inventory_digest")
        object.__setattr__(
            self,
            "ground_point_count",
            _require_nonnegative_int(self.ground_point_count, "ground_point_count"),
        )
        for name in (
            "supported_point_count",
            "unsupported_point_count",
            "negative_tolerance_exceeded_point_count",
            "hag_valid_positive_count",
            "hag_valid_near_zero_count",
            "hag_within_negative_tolerance_count",
            "hag_missing_count",
            "hag_invalid_count",
            "hag_out_of_domain_count",
        ):
            object.__setattr__(
                self, name, _require_nonnegative_int(getattr(self, name), name)
            )
        _require_exact_str(self.coordinate_frame, "coordinate_frame")
        if self.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
            raise ValueError("coordinate_frame must be LOCAL_CARTESIAN_XYZ")
        _require_exact_str(self.units, "units")
        if self.units != "m":
            raise ValueError("units must be m")
        object.__setattr__(self, "origin_xyz", _validate_vec3(self.origin_xyz, "origin_xyz"))
        _require_exact_str(self.transform_declaration, "transform_declaration")
        if self.transform_declaration != "identity":
            raise ValueError("transform_declaration must be identity")
        object.__setattr__(
            self,
            "pass_through_dimensions",
            _validate_enum_tuple(
                self.pass_through_dimensions,
                FrvSourcePassThroughDimension,
                "pass_through_dimensions",
            ),
        )
        if FrvSourcePassThroughDimension.CLASSIFICATION not in self.pass_through_dimensions:
            raise ValueError("pass_through_dimensions must include classification")
        if type(self.quality_flags) is not int or type(self.reason_flags) is not int:
            raise ValueError("quality_flags and reason_flags must be built-in ints")
        if self.quality_flags != 0 or self.reason_flags != 0:
            raise ValueError("success proof must have NONE quality and reason flags")
        error_codes = _require_tuple(self.error_codes, "error_codes")
        for code in error_codes:
            _require_enum(code, FrvErrorCode, "error_codes[]")
        if error_codes != ():
            raise ValueError("success proof must have empty error_codes")
        _require_sha(self.proof_digest, "proof_digest")
        if self.proof_digest != self.computed_digest():
            raise ValueError("proof_digest does not match the canonical P-HAG proof fields")

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": FRV_HAG_PROOF_DIGEST_DOMAIN,
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_role": self.provider_role.value,
            "interface_version": self.interface_version,
            "algorithm_version": self.algorithm_version,
            "configuration_digest": self.configuration_digest,
            "provider_config_digest": self.provider_config_digest,
            "source_artifact_reference_id": self.source_artifact_reference_id,
            "artifact_reference_id": self.artifact_reference_id,
            "artifact_components": [
                component.to_dict() for component in self.artifact_components
            ],
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "source_proof_digest": self.source_proof_digest,
            "source_binding_digest": self.source_binding_digest,
            "source_point_count": self.source_point_count,
            "domain_kind": self.domain_kind.value,
            "domain_count": self.domain_count,
            "original_index_domain_digest": self.original_index_domain_digest,
            "hag_point_content_digest": self.hag_point_content_digest,
            "ground_cell_inventory_digest": self.ground_cell_inventory_digest,
            "ground_point_count": self.ground_point_count,
            "supported_point_count": self.supported_point_count,
            "unsupported_point_count": self.unsupported_point_count,
            "negative_tolerance_exceeded_point_count": (
                self.negative_tolerance_exceeded_point_count
            ),
            "hag_valid_positive_count": self.hag_valid_positive_count,
            "hag_valid_near_zero_count": self.hag_valid_near_zero_count,
            "hag_within_negative_tolerance_count": (
                self.hag_within_negative_tolerance_count
            ),
            "hag_missing_count": self.hag_missing_count,
            "hag_invalid_count": self.hag_invalid_count,
            "hag_out_of_domain_count": self.hag_out_of_domain_count,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_xyz": list(self.origin_xyz),
            "transform_declaration": self.transform_declaration,
            "pass_through_dimensions": [
                dimension.value for dimension in self.pass_through_dimensions
            ],
            "quality_flags": self.quality_flags,
            "reason_flags": self.reason_flags,
            "error_codes": [code.value for code in self.error_codes],
        }

    def computed_digest(self) -> str:
        return frv_digest(self.to_digest_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.to_digest_dict())
        payload.pop("digest_domain")
        payload["proof_digest"] = self.proof_digest
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvHagProviderProof:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_role"] = _enum_from_value(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        values["domain_kind"] = _enum_from_value(
            values["domain_kind"], FrvA2DomainKind, "domain_kind"
        )
        values["artifact_components"] = tuple(
            FrvA2ArtifactComponentBinding.from_dict(item)
            for item in _tuple_from_json(values["artifact_components"], "artifact_components")
        )
        values["origin_xyz"] = _tuple_from_json(values["origin_xyz"], "origin_xyz")
        values["pass_through_dimensions"] = tuple(
            _enum_from_value(
                item, FrvSourcePassThroughDimension, "pass_through_dimensions[]"
            )
            for item in _tuple_from_json(
                values["pass_through_dimensions"], "pass_through_dimensions"
            )
        )
        values["error_codes"] = tuple(
            _enum_from_value(item, FrvErrorCode, "error_codes[]")
            for item in _tuple_from_json(values["error_codes"], "error_codes")
        )
        return cls(**values)


def _validate_proof_structure(proof: FrvHagProviderProof) -> None:
    if type(proof) is not FrvHagProviderProof:
        raise ValueError("proof must be a FrvHagProviderProof")
    if proof.schema_version != FRV_HAG_PROOF_SCHEMA_VERSION:
        raise ValueError("schema_version must match the frozen proof schema version")
    if proof.provider_key != FRV_HAG_PROVIDER_KEY:
        raise ValueError("provider_key must be hag_provider")
    if type(proof.provider_role) is not FrvA2ProviderRole:
        raise ValueError("provider_role must be a FrvA2ProviderRole")
    if proof.provider_role is not FrvA2ProviderRole.HAG:
        raise ValueError("provider_role must be HAG")
    if proof.interface_version != FRV_HAG_PROVIDER_INTERFACE_VERSION:
        raise ValueError("interface_version must match the frozen interface version")
    if proof.algorithm_version != FRV_HAG_ALGORITHM_VERSION:
        raise ValueError("algorithm_version must match the frozen algorithm version")
    _require_sha(proof.configuration_digest, "configuration_digest")
    _require_sha(proof.provider_config_digest, "provider_config_digest")
    _require_artifact_reference_id(
        proof.source_artifact_reference_id, "source_artifact_reference_id"
    )
    _require_artifact_reference_id(proof.artifact_reference_id, "artifact_reference_id")
    components = _require_tuple(proof.artifact_components, "artifact_components")
    for component in components:
        if type(component) is not FrvA2ArtifactComponentBinding:
            raise ValueError("artifact_components must contain FrvA2ArtifactComponentBinding")
    expected_reference = _derive_artifact_reference_id(
        proof.provider_role, proof.provider_key, components
    )
    if proof.artifact_reference_id != expected_reference:
        raise ValueError("artifact_reference_id does not match the canonical artifact components")
    _require_sha(proof.source_content_identity_digest, "source_content_identity_digest")
    _require_sha(proof.source_identity_record_digest, "source_identity_record_digest")
    _require_sha(proof.source_proof_digest, "source_proof_digest")
    _require_sha(proof.source_binding_digest, "source_binding_digest")
    if type(proof.source_point_count) is not int or proof.source_point_count <= 0:
        raise ValueError("source_point_count must be a positive built-in int")
    if type(proof.domain_count) is not int or proof.domain_count <= 0:
        raise ValueError("domain_count must be a positive built-in int")
    if proof.domain_count != proof.source_point_count:
        raise ValueError("domain_count must equal source_point_count")
    if type(proof.domain_kind) is not FrvA2DomainKind:
        raise ValueError("domain_kind must be a FrvA2DomainKind")
    if proof.domain_kind is not FrvA2DomainKind.FULL_SOURCE:
        raise ValueError("domain_kind must be FULL_SOURCE")
    _require_sha(proof.original_index_domain_digest, "original_index_domain_digest")
    _require_sha(proof.hag_point_content_digest, "hag_point_content_digest")
    _require_sha(proof.ground_cell_inventory_digest, "ground_cell_inventory_digest")
    for name in (
        "ground_point_count",
        "supported_point_count",
        "unsupported_point_count",
        "negative_tolerance_exceeded_point_count",
        "hag_valid_positive_count",
        "hag_valid_near_zero_count",
        "hag_within_negative_tolerance_count",
        "hag_missing_count",
        "hag_invalid_count",
        "hag_out_of_domain_count",
    ):
        if type(getattr(proof, name)) is not int or getattr(proof, name) < 0:
            raise ValueError(f"{name} must be a nonnegative built-in int")
    if proof.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
        raise ValueError("coordinate_frame must be LOCAL_CARTESIAN_XYZ")
    if proof.units != "m":
        raise ValueError("units must be m")
    _validate_vec3(proof.origin_xyz, "origin_xyz")
    if proof.transform_declaration != "identity":
        raise ValueError("transform_declaration must be identity")
    dimensions = _validate_enum_tuple(
        proof.pass_through_dimensions,
        FrvSourcePassThroughDimension,
        "pass_through_dimensions",
    )
    if FrvSourcePassThroughDimension.CLASSIFICATION not in dimensions:
        raise ValueError("pass_through_dimensions must include classification")
    if proof.quality_flags != 0 or proof.reason_flags != 0:
        raise ValueError("success proof must have NONE quality and reason flags")
    error_codes = _require_tuple(proof.error_codes, "error_codes")
    for code in error_codes:
        _require_enum(code, FrvErrorCode, "error_codes[]")
    if error_codes != ():
        raise ValueError("success proof must have empty error_codes")
    _require_sha(proof.proof_digest, "proof_digest")
    if proof.proof_digest != proof.computed_digest():
        raise ValueError("proof_digest does not match the canonical P-HAG proof fields")


def validate_frv_hag_provider_proof(proof: FrvHagProviderProof) -> None:
    _validate_proof_structure(proof)
    if proof.supported_point_count + proof.unsupported_point_count != proof.source_point_count:
        raise ValueError("supported and unsupported counts must sum to source_point_count")
    state_sum = (
        proof.hag_valid_positive_count
        + proof.hag_valid_near_zero_count
        + proof.hag_within_negative_tolerance_count
        + proof.hag_missing_count
        + proof.hag_invalid_count
        + proof.hag_out_of_domain_count
    )
    if state_sum != proof.source_point_count:
        raise ValueError("HAG state counts must sum to source_point_count")
    if proof.hag_missing_count != proof.unsupported_point_count:
        raise ValueError("hag_missing_count must equal unsupported_point_count")
    if proof.negative_tolerance_exceeded_point_count != proof.hag_invalid_count:
        raise ValueError(
            "negative_tolerance_exceeded_point_count must equal hag_invalid_count"
        )
    if proof.hag_out_of_domain_count != 0:
        raise ValueError("hag_out_of_domain_count must be zero for a FULL_SOURCE proof")
    expected_index_digest = _stream_index_domain_digest(
        proof.domain_count, proof.source_content_identity_digest
    )
    if proof.original_index_domain_digest != expected_index_digest:
        raise ValueError("original_index_domain_digest does not match the FULL_SOURCE domain")
    if proof.proof_digest != proof.computed_digest():
        raise ValueError("proof_digest does not match the canonical P-HAG proof fields")


# ---------------------------------------------------------------------------
# Finalized P-HAG result with private capability
# ---------------------------------------------------------------------------

_MISSING = object()


class _FRV_HAG_FINALIZED_CAPABILITY:
    """Module-private identity-only finalized P-HAG capability token."""

    __slots__ = ()


@dataclass(frozen=True, slots=True, init=False)
class FrvHagProviderResult:
    """Non-serializable finalized P-HAG success wrapper (private capability)."""

    proof: FrvHagProviderProof
    source_identity: SourceIdentity
    _finalized_capability: Any = field(init=False, repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrvHagProviderResult has no public constructor")


def _validate_finalized_hag_bindings(
    proof: FrvHagProviderProof, source_identity: SourceIdentity
) -> None:
    validate_frv_hag_provider_proof(proof)
    validate_source_identity_record_digest(source_identity)
    validate_source_content_identity_digest(source_identity)
    if proof.source_content_identity_digest != source_identity.computed_content_digest():
        raise ValueError("source_content_identity_digest must match the computed source identity")
    if proof.source_identity_record_digest != source_identity.computed_record_digest():
        raise ValueError("source_identity_record_digest must match the computed source identity")
    if len(proof.artifact_components) != 1:
        raise ValueError("exactly one artifact component is required")
    if proof.artifact_components[0].sha256 != source_identity.source_file_sha256:
        raise ValueError("artifact component sha256 must match the source identity")


def _create_finalized_frv_hag_provider_result(
    proof: FrvHagProviderProof,
    source_identity: SourceIdentity,
    *,
    capability: Any,
) -> FrvHagProviderResult:
    if type(proof) is not FrvHagProviderProof:
        raise TypeError("proof must be a FrvHagProviderProof")
    if type(source_identity) is not SourceIdentity:
        raise TypeError("source_identity must be a SourceIdentity")
    if capability is not _FRV_HAG_FINALIZED_CAPABILITY:
        raise ValueError("capability must be the module-private finalized capability")
    _validate_finalized_hag_bindings(proof, source_identity)
    result = object.__new__(FrvHagProviderResult)
    object.__setattr__(result, "proof", proof)
    object.__setattr__(result, "source_identity", source_identity)
    object.__setattr__(result, "_finalized_capability", capability)
    return result


def validate_finalized_frv_hag_provider_result(result: FrvHagProviderResult) -> None:
    if type(result) is not FrvHagProviderResult:
        raise TypeError("result must be a FrvHagProviderResult")
    if getattr(result, "_finalized_capability", _MISSING) is not (
        _FRV_HAG_FINALIZED_CAPABILITY
    ):
        raise ValueError("result must carry the module-private finalized capability")
    proof = getattr(result, "proof", _MISSING)
    if type(proof) is not FrvHagProviderProof:
        raise ValueError("result.proof must be a FrvHagProviderProof")
    source_identity = getattr(result, "source_identity", _MISSING)
    if type(source_identity) is not SourceIdentity:
        raise ValueError("result.source_identity must be a SourceIdentity")
    _validate_finalized_hag_bindings(proof, source_identity)


# ---------------------------------------------------------------------------
# FrvHagLoadError
# ---------------------------------------------------------------------------


class FrvHagLoadError(Exception):
    """P-HAG load failure carrying a frozen A2 failure record."""

    def __init__(
        self,
        *,
        failure_record: FrvA2FailureRecord,
        quality_flags: int,
    ) -> None:
        if type(failure_record) is not FrvA2FailureRecord:
            raise TypeError("failure_record must be a FrvA2FailureRecord")
        validate_frv_a2_failure_record(failure_record)
        if type(quality_flags) is not int:
            raise TypeError("quality_flags must be a built-in int")
        if quality_flags < 0:
            raise ValueError("quality_flags must be nonnegative")
        validate_frv_flag_mask(quality_flags, FrvQualityFlag, "quality_flags")
        self.failure_record = failure_record
        self.quality_flags = quality_flags
        super().__init__(failure_record.detail)


# ---------------------------------------------------------------------------
# FrvHagLoadSession protocol
# ---------------------------------------------------------------------------


class FrvHagLoadSession(Protocol):
    """Single-use P-HAG load session interface (contracts only)."""

    def iter_chunks(self) -> Iterator[FrvHagChunk]: ...
    def finalize(self) -> FrvHagProviderResult: ...
    def close(self) -> None: ...
    def __enter__(self) -> FrvHagLoadSession: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]: ...


# ---------------------------------------------------------------------------
# FRVA2BND1 adapter
# ---------------------------------------------------------------------------


def to_frv_a2_hag_provider_binding_evidence(
    result: FrvHagProviderResult,
) -> FrvA2ProviderBindingEvidence:
    """Return the success-only FRVA2BND1 formal provider evidence for P-HAG."""

    validate_finalized_frv_hag_provider_result(result)
    proof = result.proof
    evidence = FrvA2ProviderBindingEvidence.__new__(FrvA2ProviderBindingEvidence)
    object.__setattr__(evidence, "provider_key", FRV_HAG_PROVIDER_KEY)
    object.__setattr__(evidence, "provider_role", FrvA2ProviderRole.HAG)
    object.__setattr__(evidence, "evidence_class", FrvA2EvidenceClass.FORMAL_PROVIDER)
    object.__setattr__(evidence, "public_module", "algorithms.vegetation.hag_provider")
    object.__setattr__(evidence, "public_type_name", "FrvHagProviderResult")
    object.__setattr__(evidence, "public_loader_name", "load_frv_hag")
    object.__setattr__(evidence, "interface_version", FRV_HAG_PROVIDER_INTERFACE_VERSION)
    object.__setattr__(evidence, "schema_version", FRV_HAG_PROVIDER_SCHEMA_VERSION)
    object.__setattr__(evidence, "algorithm_version", FRV_HAG_ALGORITHM_VERSION)
    object.__setattr__(evidence, "configuration_digest", proof.configuration_digest)
    object.__setattr__(evidence, "provider_config_digest", proof.provider_config_digest)
    object.__setattr__(evidence, "artifact_reference_id", proof.artifact_reference_id)
    object.__setattr__(evidence, "artifact_components", proof.artifact_components)
    object.__setattr__(
        evidence, "source_content_identity_digest", proof.source_content_identity_digest
    )
    object.__setattr__(
        evidence, "source_identity_record_digest", proof.source_identity_record_digest
    )
    object.__setattr__(evidence, "coordinate_frame", "LOCAL_CARTESIAN_XYZ")
    object.__setattr__(evidence, "units", "m")
    object.__setattr__(evidence, "origin_xyz", proof.origin_xyz)
    object.__setattr__(evidence, "transform_declaration", "identity")
    object.__setattr__(evidence, "domain_kind", FrvA2DomainKind.FULL_SOURCE)
    object.__setattr__(evidence, "domain_count", proof.domain_count)
    object.__setattr__(
        evidence, "original_index_domain_digest", proof.original_index_domain_digest
    )
    object.__setattr__(evidence, "sample_index_domain_digest", None)
    object.__setattr__(evidence, "provider_readiness", FrvA2ProviderReadiness.VALIDATED)
    object.__setattr__(evidence, "quality_flags", 0)
    object.__setattr__(evidence, "reason_flags", 0)
    object.__setattr__(evidence, "error_codes", ())
    upstream = tuple(
        sorted(
            {
                proof.ground_cell_inventory_digest,
                proof.hag_point_content_digest,
                proof.source_binding_digest,
                proof.source_proof_digest,
            }
        )
    )
    object.__setattr__(evidence, "upstream_binding_digests", upstream)
    object.__setattr__(evidence, "review_codes", ())
    object.__setattr__(evidence, "limitation_codes", ())
    object.__setattr__(evidence, "binding_digest", frv_a2_provider_binding_digest(evidence))
    return evidence


__all__ = [
    "FRV_HAG_PROVIDER_KEY",
    "FRV_HAG_PROVIDER_INTERFACE_VERSION",
    "FRV_HAG_PROVIDER_SCHEMA_VERSION",
    "FRV_HAG_CONFIG_SCHEMA_VERSION",
    "FRV_HAG_PROOF_SCHEMA_VERSION",
    "FRV_HAG_ALGORITHM_VERSION",
    "FRV_HAG_CONFIG_DIGEST_DOMAIN",
    "FRV_HAG_SOURCE_PROOF_DIGEST_DOMAIN",
    "FRV_HAG_GRID_DIGEST_DOMAIN",
    "FRV_HAG_POINT_CONTENT_DIGEST_DOMAIN",
    "FRV_HAG_PROOF_DIGEST_DOMAIN",
    "FrvHagProviderConfig",
    "FrvHagChunk",
    "FrvHagProviderProof",
    "FrvHagProviderResult",
    "FrvHagLoadError",
    "FrvHagLoadSession",
    "validate_frv_hag_provider_config",
    "validate_frv_hag_chunk",
    "validate_frv_hag_provider_proof",
    "validate_finalized_frv_hag_provider_result",
    "derive_frv_hag_provider_config_digest",
    "frv_hag_ground_cell_inventory_digest",
    "frv_hag_point_content_digest",
    "to_frv_a2_hag_provider_binding_evidence",
]
