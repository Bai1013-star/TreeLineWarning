"""FRV-A2 P-SOURCE formal source point-cloud provider contracts (C1).

C1 establishes the frozen public constants, the closed P-SOURCE enums, and the
exact-shape immutable contracts consumed by the P-SOURCE provider and its
downstream stages.  This module deliberately implements no file access, no
laspy session, no streaming digest, no failure mapping, and no A2 BND1 adapter.
It also does not recompute ``provider_config_digest`` (``FRVSRCFG1`` is a later
stage) and performs no cross-object reference validation.

``FrvSourceProviderReference`` and ``FrvSourceChunk`` are non-serializable
runtime objects: they provide no ``to_dict``/``from_dict`` and must never enter
canonical FRV JSON serialization.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from core.contracts.frv import (
    FrvCanonicalPointContentSpec,
    FrvQualityFlag,
    SourceIdentity,
    canonical_frv_json_bytes,
    frv_digest,
    validate_frv_flag_mask,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2DomainKind,
    FrvA2EvidenceClass,
    FrvA2FailureRecord,
    FrvA2ProviderBindingEvidence,
    FrvA2ProviderReadiness,
    FrvA2ProviderRole,
    _derive_artifact_reference_id,
    frv_a2_index_domain_digest,
    frv_a2_provider_binding_digest,
    validate_frv_a2_failure_record,
)

# ---------------------------------------------------------------------------
# Public constants (frozen names and values)
# ---------------------------------------------------------------------------

FRV_SOURCE_PROVIDER_KEY = "source_scan"
FRV_SOURCE_PROVIDER_INTERFACE_VERSION = "frv-source-provider-v1"
FRV_SOURCE_PROVIDER_SCHEMA_VERSION = "frv-source-provider-result-v1"
FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION = "frv-source-provider-config-v1"
FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION = "frv-source-header-v1"
FRV_SOURCE_PROVIDER_ALGORITHM_VERSION = "frv-source-laspy-single-handle-v1"
FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN = "FRVSRCFG1"
FRV_SOURCE_DEFAULT_CHUNK_SIZE = 1000000

_FRV_SOURCE_ARTIFACT_REFERENCE_PREFIX = "frv-a2-ref-v1|"

# ---------------------------------------------------------------------------
# Public closed enums
# ---------------------------------------------------------------------------


class FrvSourceModel(str, Enum):
    SINGLE_FILE_ONLY = "single_file_only"


class FrvSourceCoordinateConversionPolicy(str, Enum):
    FORBID = "forbid"


class FrvSourceInvalidPointPolicy(str, Enum):
    FAIL_CLOSED = "fail_closed"


class FrvSourceCanonicalContentPolicy(str, Enum):
    VERIFY_IF_DECLARED = "verify_if_declared"


class FrvSourceFormat(str, Enum):
    LAS = "las"


class FrvSourceCompressionState(str, Enum):
    UNCOMPRESSED = "uncompressed"


class FrvSourceCrsAuthorityState(str, Enum):
    ABSENT = "absent"


class FrvSourceVlrPolicyState(str, Enum):
    NO_CRS_AUTHORITY_DETECTED = "no_crs_authority_detected"


class FrvSourcePassThroughDimension(str, Enum):
    CLASSIFICATION = "classification"
    INTENSITY = "intensity"
    RETURN_NUMBER = "return_number"
    NUMBER_OF_RETURNS = "number_of_returns"
    SCAN_ANGLE = "scan_angle"
    GPS_TIME = "gps_time"
    POINT_SOURCE_ID = "point_source_id"
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


# Chunk pass-through attribute dtype and finiteness rules keyed by dimension.
_FRV_SOURCE_ATTRIBUTE_DTYPES: dict[FrvSourcePassThroughDimension, np.dtype] = {
    FrvSourcePassThroughDimension.CLASSIFICATION: np.dtype("|u1"),
    FrvSourcePassThroughDimension.INTENSITY: np.dtype("<u2"),
    FrvSourcePassThroughDimension.RETURN_NUMBER: np.dtype("|u1"),
    FrvSourcePassThroughDimension.NUMBER_OF_RETURNS: np.dtype("|u1"),
    FrvSourcePassThroughDimension.SCAN_ANGLE: np.dtype("<f8"),
    FrvSourcePassThroughDimension.GPS_TIME: np.dtype("<f8"),
    FrvSourcePassThroughDimension.POINT_SOURCE_ID: np.dtype("<u2"),
    FrvSourcePassThroughDimension.RED: np.dtype("<u2"),
    FrvSourcePassThroughDimension.GREEN: np.dtype("<u2"),
    FrvSourcePassThroughDimension.BLUE: np.dtype("<u2"),
}
_FRV_SOURCE_ATTRIBUTE_FINITE_REQUIRED = frozenset(
    {
        FrvSourcePassThroughDimension.SCAN_ANGLE,
        FrvSourcePassThroughDimension.GPS_TIME,
    }
)

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


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a nonnegative built-in int")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be a positive built-in int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_finite_float(value: Any, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite built-in float")
    return value


def _require_enum(value: Any, enum_type: type[Enum], name: str) -> Enum:
    if type(value) is not enum_type:
        raise ValueError(f"{name} must be a {enum_type.__name__}")
    return value


def _require_tuple(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) is not tuple:
        raise ValueError(f"{name} must be an immutable tuple")
    return value


def _tuple_from_json(value: Any, name: str) -> tuple[Any, ...]:
    if type(value) not in (list, tuple):
        raise ValueError(f"{name} must be a JSON array")
    return tuple(value)


def _enum_from_value(value: Any, enum_type: type[Enum], name: str) -> Enum:
    text = _require_exact_str(value, name)
    try:
        return enum_type(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is unsupported") from exc


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


def _enum_order(enum_type: type[Enum]) -> dict[Enum, int]:
    return {member: index for index, member in enumerate(enum_type)}


def _validate_enum_tuple(
    value: Any, enum_type: type[Enum], name: str
) -> tuple[Enum, ...]:
    items = _require_tuple(value, name)
    order = _enum_order(enum_type)
    seen: set[Enum] = set()
    for item in items:
        _require_enum(item, enum_type, f"{name}[]")
        if item in seen:
            raise ValueError(f"{name} must be unique")
        seen.add(item)
    if tuple(sorted(items, key=order.__getitem__)) != items:
        raise ValueError(f"{name} must use declaration order")
    return items


def _validate_vec3(value: Any, name: str) -> tuple[float, float, float]:
    items = _require_tuple(value, name)
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return (
        _require_finite_float(items[0], f"{name}[0]"),
        _require_finite_float(items[1], f"{name}[1]"),
        _require_finite_float(items[2], f"{name}[2]"),
    )


def _validate_dimension_names(value: Any, name: str) -> tuple[str, ...]:
    items = _require_tuple(value, name)
    validated = tuple(_require_exact_str(item, f"{name}[]") for item in items)
    for item in validated:
        if item != item.lower():
            raise ValueError(f"{name}[] must be lower-case")
    if len(set(validated)) != len(validated):
        raise ValueError(f"{name} must be unique")
    if tuple(sorted(validated)) != validated:
        raise ValueError(f"{name} must be lexicographically sorted")
    return validated


def _require_artifact_reference_id(value: Any, name: str) -> str:
    """Require a real A2 ``frv-a2-ref-v1|...`` reference string, unchanged.

    The A2 artifact reference grammar is authoritative; the supplied value is
    kept verbatim (no SHA validation, no re-derivation, no auto-normalization).
    """

    text = _require_exact_str(value, name)
    if not text.startswith(_FRV_SOURCE_ARTIFACT_REFERENCE_PREFIX):
        raise ValueError(f"{name} must start with {_FRV_SOURCE_ARTIFACT_REFERENCE_PREFIX!r}")
    return text


def _validate_ndarray(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool,
) -> np.ndarray:
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
    return value


def _frozen_ndarray(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool,
) -> np.ndarray:
    """Validate then copy to an immutable bytes-backed read-only array.

    The returned array shares an immutable ``bytes`` buffer, so the external
    caller cannot re-enable writing via ``setflags(write=True)`` and cannot
    mutate the stored chunk through an alias.
    """

    validated = _validate_ndarray(
        value, name, dtype=dtype, shape=shape, finite=finite
    )
    payload = validated.tobytes(order="C")
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


# ---------------------------------------------------------------------------
# FRVSRCFG1 provider-config digest
# ---------------------------------------------------------------------------


def derive_frv_source_provider_config_digest(
    *,
    schema_version: str,
    source_model: FrvSourceModel,
    coordinate_conversion_policy: FrvSourceCoordinateConversionPolicy,
    invalid_point_policy: FrvSourceInvalidPointPolicy,
    canonical_content_policy: FrvSourceCanonicalContentPolicy,
    pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...],
) -> str:
    """Return the FRVSRCFG1 digest for the frozen provider-config payload.

    The payload uses the exact frozen framing from the P-SOURCE plan: it never
    includes ``provider_config_digest``, chunk size, paths, or runtime values,
    and pass-through dimensions keep their frozen declaration order.
    """

    _require_exact_str(schema_version, "schema_version")
    _require_enum(source_model, FrvSourceModel, "source_model")
    _require_enum(
        coordinate_conversion_policy,
        FrvSourceCoordinateConversionPolicy,
        "coordinate_conversion_policy",
    )
    _require_enum(
        invalid_point_policy, FrvSourceInvalidPointPolicy, "invalid_point_policy"
    )
    _require_enum(
        canonical_content_policy,
        FrvSourceCanonicalContentPolicy,
        "canonical_content_policy",
    )
    dimensions = _validate_enum_tuple(
        pass_through_dimensions, FrvSourcePassThroughDimension, "pass_through_dimensions"
    )
    payload = {
        "digest_domain": "FRVSRCFG1",
        "schema_version": schema_version,
        "source_model": source_model.value,
        "coordinate_conversion_policy": coordinate_conversion_policy.value,
        "invalid_point_policy": invalid_point_policy.value,
        "canonical_content_policy": canonical_content_policy.value,
        "pass_through_dimensions": [dimension.value for dimension in dimensions],
    }
    return frv_digest(payload)


# ---------------------------------------------------------------------------
# FrvSourceProviderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvSourceProviderConfig:
    """Exact-schema P-SOURCE provider configuration (digest domain FRVSRCFG1).

    The ``provider_config_digest`` is validated as a strict SHA-256 string;
    the ``FRVSRCFG1`` recomputation belongs to a later stage.
    """

    schema_version: str
    source_model: FrvSourceModel
    coordinate_conversion_policy: FrvSourceCoordinateConversionPolicy
    invalid_point_policy: FrvSourceInvalidPointPolicy
    canonical_content_policy: FrvSourceCanonicalContentPolicy
    pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...]
    provider_config_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported P-SOURCE provider config schema version")
        _require_enum(self.source_model, FrvSourceModel, "source_model")
        _require_enum(
            self.coordinate_conversion_policy,
            FrvSourceCoordinateConversionPolicy,
            "coordinate_conversion_policy",
        )
        _require_enum(
            self.invalid_point_policy, FrvSourceInvalidPointPolicy, "invalid_point_policy"
        )
        _require_enum(
            self.canonical_content_policy,
            FrvSourceCanonicalContentPolicy,
            "canonical_content_policy",
        )
        object.__setattr__(
            self,
            "pass_through_dimensions",
            _validate_enum_tuple(
                self.pass_through_dimensions,
                FrvSourcePassThroughDimension,
                "pass_through_dimensions",
            ),
        )
        _require_sha(self.provider_config_digest, "provider_config_digest")
        computed = derive_frv_source_provider_config_digest(
            schema_version=self.schema_version,
            source_model=self.source_model,
            coordinate_conversion_policy=self.coordinate_conversion_policy,
            invalid_point_policy=self.invalid_point_policy,
            canonical_content_policy=self.canonical_content_policy,
            pass_through_dimensions=self.pass_through_dimensions,
        )
        if self.provider_config_digest != computed:
            raise ValueError("provider_config_digest does not match canonical FRVSRCFG1 fields")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_model": self.source_model.value,
            "coordinate_conversion_policy": self.coordinate_conversion_policy.value,
            "invalid_point_policy": self.invalid_point_policy.value,
            "canonical_content_policy": self.canonical_content_policy.value,
            "pass_through_dimensions": [
                dimension.value for dimension in self.pass_through_dimensions
            ],
            "provider_config_digest": self.provider_config_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvSourceProviderConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["source_model"] = _enum_from_value(
            values["source_model"], FrvSourceModel, "source_model"
        )
        values["coordinate_conversion_policy"] = _enum_from_value(
            values["coordinate_conversion_policy"],
            FrvSourceCoordinateConversionPolicy,
            "coordinate_conversion_policy",
        )
        values["invalid_point_policy"] = _enum_from_value(
            values["invalid_point_policy"], FrvSourceInvalidPointPolicy, "invalid_point_policy"
        )
        values["canonical_content_policy"] = _enum_from_value(
            values["canonical_content_policy"],
            FrvSourceCanonicalContentPolicy,
            "canonical_content_policy",
        )
        values["pass_through_dimensions"] = tuple(
            _enum_from_value(
                item, FrvSourcePassThroughDimension, "pass_through_dimensions[]"
            )
            for item in _tuple_from_json(
                values["pass_through_dimensions"], "pass_through_dimensions"
            )
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# FrvSourceHeaderMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvSourceHeaderMetadata:
    """Exact-schema LAS header metadata captured from the provider reader."""

    schema_version: str
    source_format: FrvSourceFormat
    las_version: str
    point_format_id: int
    point_record_length_bytes: int
    point_count: int
    scale_xyz: tuple[float, float, float]
    offset_xyz: tuple[float, float, float]
    min_xyz: tuple[float, float, float]
    max_xyz: tuple[float, float, float]
    dimensions: tuple[str, ...]
    extra_dimensions: tuple[str, ...]
    compression_state: FrvSourceCompressionState
    crs_authority_state: FrvSourceCrsAuthorityState
    non_crs_vlr_count: int
    non_crs_evlr_count: int
    vlr_policy_state: FrvSourceVlrPolicyState

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION:
            raise ValueError("unsupported P-SOURCE header schema version")
        _require_enum(self.source_format, FrvSourceFormat, "source_format")
        _require_exact_str(self.las_version, "las_version")
        _require_nonnegative_int(self.point_format_id, "point_format_id")
        _require_positive_int(self.point_record_length_bytes, "point_record_length_bytes")
        _require_nonnegative_int(self.point_count, "point_count")
        _require_enum(self.compression_state, FrvSourceCompressionState, "compression_state")
        _require_enum(
            self.crs_authority_state, FrvSourceCrsAuthorityState, "crs_authority_state"
        )
        _require_nonnegative_int(self.non_crs_vlr_count, "non_crs_vlr_count")
        _require_nonnegative_int(self.non_crs_evlr_count, "non_crs_evlr_count")
        _require_enum(self.vlr_policy_state, FrvSourceVlrPolicyState, "vlr_policy_state")
        object.__setattr__(self, "scale_xyz", _validate_vec3(self.scale_xyz, "scale_xyz"))
        object.__setattr__(self, "offset_xyz", _validate_vec3(self.offset_xyz, "offset_xyz"))
        object.__setattr__(self, "min_xyz", _validate_vec3(self.min_xyz, "min_xyz"))
        object.__setattr__(self, "max_xyz", _validate_vec3(self.max_xyz, "max_xyz"))
        object.__setattr__(
            self, "dimensions", _validate_dimension_names(self.dimensions, "dimensions")
        )
        object.__setattr__(
            self,
            "extra_dimensions",
            _validate_dimension_names(self.extra_dimensions, "extra_dimensions"),
        )
        if set(self.extra_dimensions) & set(self.dimensions):
            raise ValueError("extra_dimensions must not overlap dimensions")
        for min_value, max_value in zip(self.min_xyz, self.max_xyz, strict=True):
            if min_value > max_value:
                raise ValueError("min_xyz must not exceed max_xyz on any axis")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_format": self.source_format.value,
            "las_version": self.las_version,
            "point_format_id": self.point_format_id,
            "point_record_length_bytes": self.point_record_length_bytes,
            "point_count": self.point_count,
            "scale_xyz": list(self.scale_xyz),
            "offset_xyz": list(self.offset_xyz),
            "min_xyz": list(self.min_xyz),
            "max_xyz": list(self.max_xyz),
            "dimensions": list(self.dimensions),
            "extra_dimensions": list(self.extra_dimensions),
            "compression_state": self.compression_state.value,
            "crs_authority_state": self.crs_authority_state.value,
            "non_crs_vlr_count": self.non_crs_vlr_count,
            "non_crs_evlr_count": self.non_crs_evlr_count,
            "vlr_policy_state": self.vlr_policy_state.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvSourceHeaderMetadata:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["source_format"] = _enum_from_value(
            values["source_format"], FrvSourceFormat, "source_format"
        )
        for name in ("scale_xyz", "offset_xyz", "min_xyz", "max_xyz"):
            values[name] = _tuple_from_json(values[name], name)
        values["dimensions"] = _tuple_from_json(values["dimensions"], "dimensions")
        values["extra_dimensions"] = _tuple_from_json(
            values["extra_dimensions"], "extra_dimensions"
        )
        values["compression_state"] = _enum_from_value(
            values["compression_state"], FrvSourceCompressionState, "compression_state"
        )
        values["crs_authority_state"] = _enum_from_value(
            values["crs_authority_state"], FrvSourceCrsAuthorityState, "crs_authority_state"
        )
        values["vlr_policy_state"] = _enum_from_value(
            values["vlr_policy_state"], FrvSourceVlrPolicyState, "vlr_policy_state"
        )
        return cls(**values)


# ---------------------------------------------------------------------------
# FrvSourceProviderReference (non-serializable runtime object)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvSourceProviderReference:
    """Non-serializable runtime binding between A1 identity and an access path.

    C1 validates local shape and types only; cross-object reference validation
    (role/key values, component layout, path binding, digest recomputation)
    belongs to a later reference-validator stage.
    """

    provider_role: FrvA2ProviderRole
    provider_key: str
    artifact_reference_id: str
    artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    runtime_access_path: Path
    source_identity: SourceIdentity
    canonical_point_content_spec: FrvCanonicalPointContentSpec | None
    configuration_digest: str
    provider_config_digest: str
    interface_version: str
    schema_version: str
    algorithm_version: str

    def __post_init__(self) -> None:
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        _require_exact_str(self.provider_key, "provider_key")
        _require_artifact_reference_id(self.artifact_reference_id, "artifact_reference_id")
        components = _require_tuple(self.artifact_components, "artifact_components")
        for index, component in enumerate(components):
            if type(component) is not FrvA2ArtifactComponentBinding:
                raise ValueError(
                    f"artifact_components[{index}] must be a FrvA2ArtifactComponentBinding"
                )
        if not isinstance(self.runtime_access_path, Path):
            raise ValueError("runtime_access_path must be a pathlib.Path instance")
        if type(self.source_identity) is not SourceIdentity:
            raise ValueError("source_identity must be a SourceIdentity")
        if self.canonical_point_content_spec is not None and type(
            self.canonical_point_content_spec
        ) is not FrvCanonicalPointContentSpec:
            raise ValueError(
                "canonical_point_content_spec must be null or a FrvCanonicalPointContentSpec"
            )
        _require_sha(self.configuration_digest, "configuration_digest")
        _require_sha(self.provider_config_digest, "provider_config_digest")
        _require_exact_str(self.interface_version, "interface_version")
        _require_exact_str(self.schema_version, "schema_version")
        _require_exact_str(self.algorithm_version, "algorithm_version")


# ---------------------------------------------------------------------------
# FrvSourceChunk (non-serializable runtime object)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvSourceChunk:
    """Non-serializable read-only chunk of the P-SOURCE point stream.

    The caller must supply owned, read-only, C-contiguous arrays that already
    satisfy the frozen dtype/shape rules.  The contract validates the input
    strictly and then stores an immutable bytes-backed copy so that neither the
    arrays nor the attributes mapping retain an external mutable alias.
    """

    source_id: str
    chunk_id: int
    start_index: int
    point_count: int
    xyz: np.ndarray
    original_index: np.ndarray
    attributes: Mapping[str, np.ndarray]
    is_last: bool

    def __post_init__(self) -> None:
        _require_exact_str(self.source_id, "source_id")
        object.__setattr__(
            self, "chunk_id", _require_nonnegative_int(self.chunk_id, "chunk_id")
        )
        object.__setattr__(
            self, "start_index", _require_nonnegative_int(self.start_index, "start_index")
        )
        point_count = _require_nonnegative_int(self.point_count, "point_count")
        object.__setattr__(self, "point_count", point_count)
        if type(self.is_last) is not bool:
            raise ValueError("is_last must be a built-in bool")
        object.__setattr__(
            self,
            "xyz",
            _frozen_ndarray(
                self.xyz,
                "xyz",
                dtype=np.dtype("<f8"),
                shape=(point_count, 3),
                finite=True,
            ),
        )
        object.__setattr__(
            self,
            "original_index",
            _frozen_ndarray(
                self.original_index,
                "original_index",
                dtype=np.dtype("<i8"),
                shape=(point_count,),
                finite=False,
            ),
        )
        attributes = self.attributes
        if type(attributes) is not MappingProxyType:
            raise ValueError("attributes must be an immutable mapping")
        dimensions: list[FrvSourcePassThroughDimension] = []
        for key in attributes:
            if type(key) is not str:
                raise ValueError("attributes keys must be built-in str")
            try:
                dimension = FrvSourcePassThroughDimension(key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unknown pass-through dimension key {key!r}") from exc
            dimensions.append(dimension)
        order = _enum_order(FrvSourcePassThroughDimension)
        indices = [order[dimension] for dimension in dimensions]
        if indices != sorted(indices):
            raise ValueError("attributes keys must follow whitelist declaration order")
        frozen_attributes: dict[str, np.ndarray] = {}
        for dimension in dimensions:
            array = attributes[dimension.value]
            frozen_attributes[dimension.value] = _frozen_ndarray(
                array,
                f"attributes[{dimension.value}]",
                dtype=_FRV_SOURCE_ATTRIBUTE_DTYPES[dimension],
                shape=(point_count,),
                finite=dimension in _FRV_SOURCE_ATTRIBUTE_FINITE_REQUIRED,
            )
        object.__setattr__(self, "attributes", MappingProxyType(frozen_attributes))


# ---------------------------------------------------------------------------
# FRVPCD1 / FRVA2IDX1 streaming digests and stream continuity
# ---------------------------------------------------------------------------

_FRV_SOURCE_EXACT_SPEC_FIELDS = (
    ("original_index", "ORIGINAL_INDEX", "<i8"),
    ("x", "X_COORDINATE", "<f8"),
    ("y", "Y_COORDINATE", "<f8"),
    ("z", "Z_COORDINATE", "<f8"),
)


def _check_exact_point_content_spec(
    canonical_point_content_spec: FrvCanonicalPointContentSpec,
) -> None:
    """Require the frozen exact four-field P-SOURCE point-content spec."""

    if type(canonical_point_content_spec) is not FrvCanonicalPointContentSpec:
        raise TypeError(
            "canonical_point_content_spec must be a FrvCanonicalPointContentSpec"
        )
    spec = canonical_point_content_spec
    if spec.row_order != "ORIGINAL_INDEX_ASCENDING":
        raise ValueError("canonical point-content spec must use ORIGINAL_INDEX_ASCENDING")
    if spec.quantization_policy != "EXACT_TYPED_VALUE":
        raise ValueError("canonical point-content spec must use EXACT_TYPED_VALUE")
    if spec.missing_value_policy != "EXPLICIT_VALIDITY_BYTE":
        raise ValueError("canonical point-content spec must use EXPLICIT_VALIDITY_BYTE")
    fields = spec.ordered_field_specs
    if len(fields) != len(_FRV_SOURCE_EXACT_SPEC_FIELDS):
        raise ValueError("canonical point-content spec must contain exactly four fields")
    for field_spec, (name, role, dtype) in zip(
        fields, _FRV_SOURCE_EXACT_SPEC_FIELDS, strict=True
    ):
        if (
            field_spec.field_name != name
            or field_spec.semantic_role != role
            or field_spec.dtype_code != dtype
            or field_spec.byte_order != "LITTLE_ENDIAN"
            or field_spec.scale != 1.0
            or field_spec.offset != 0.0
            or field_spec.required is not True
            or field_spec.null_policy != "FORBID_NULL"
        ):
            raise ValueError(
                "canonical point-content spec must be the frozen exact four-field spec"
            )


def _iter_stream_chunks(chunks: Iterable[FrvSourceChunk]):
    """Validate stream-level continuity and yield each validated chunk.

    Checks chunk_id continuity from 0, start_index continuity (each equals the
    accumulated point count), a single shared source_id, no chunk after the
    final is_last chunk, and a final is_last chunk for any non-empty stream.
    """

    expected_chunk_id = 0
    expected_start = 0
    previous_source_id: str | None = None
    chunk_count = 0
    saw_final = False
    for chunk in chunks:
        if type(chunk) is not FrvSourceChunk:
            raise TypeError("chunks must contain FrvSourceChunk values")
        if saw_final:
            raise ValueError("chunks must not appear after the final is_last chunk")
        if chunk.chunk_id != expected_chunk_id:
            raise ValueError("chunk_id values must be strictly consecutive from 0")
        if chunk.start_index != expected_start:
            raise ValueError("start_index must equal the accumulated point count")
        if previous_source_id is not None and chunk.source_id != previous_source_id:
            raise ValueError("all chunks must share one source_id")
        if (
            len(chunk.xyz) != chunk.point_count
            or len(chunk.original_index) != chunk.point_count
        ):
            raise ValueError("chunk array length must match point_count")
        expected_chunk_id += 1
        expected_start += chunk.point_count
        previous_source_id = chunk.source_id
        chunk_count += 1
        saw_final = chunk.is_last
        yield chunk
    if chunk_count > 0 and not saw_final:
        raise ValueError("non-empty streams must end with an is_last chunk")


def derive_frv_source_point_content_digest(
    chunks: Iterable[FrvSourceChunk],
    *,
    canonical_point_content_spec: FrvCanonicalPointContentSpec,
) -> str:
    """Return the FRVPCD1 streaming digest over full-resolution point content.

    The framing is the frozen ``b"FRVPCD1\\0"`` + uint64 spec-header length +
    canonical spec JSON + ordered little-endian row bytes (original_index <i8,
    x/y/z <f8), with ``-0.0`` normalized to ``+0.0``.  The digest is
    chunk-boundary invariant and never materializes the full source.
    """

    _check_exact_point_content_spec(canonical_point_content_spec)
    spec_header = canonical_frv_json_bytes(canonical_point_content_spec.to_digest_dict())
    hasher = hashlib.sha256()
    hasher.update(b"FRVPCD1\0")
    hasher.update(struct.pack("<Q", len(spec_header)))
    hasher.update(spec_header)
    for chunk in _iter_stream_chunks(chunks):
        for row in range(chunk.point_count):
            original_index = int(chunk.original_index[row])
            x = float(chunk.xyz[row, 0]) + 0.0
            y = float(chunk.xyz[row, 1]) + 0.0
            z = float(chunk.xyz[row, 2]) + 0.0
            hasher.update(struct.pack("<qddd", original_index, x, y, z))
    return hasher.hexdigest()


def derive_frv_source_original_index_digest(
    chunks: Iterable[FrvSourceChunk],
    *,
    source_content_identity_digest: str,
) -> str:
    """Return the FRVA2IDX1 streaming digest over the original-index domain.

    The framing byte-for-byte reproduces the existing A2
    ``frv_a2_index_domain_digest`` for role ORIGINAL_INDEX, domain FULL_SOURCE,
    and the global index sequence 0..N-1, without materializing the full index
    list.  The original_index values in every chunk must equal that exact
    global sequence.
    """

    _require_sha(source_content_identity_digest, "source_content_identity_digest")
    total = 0
    for chunk in _iter_stream_chunks(chunks):
        for row in range(chunk.point_count):
            if int(chunk.original_index[row]) != chunk.start_index + row:
                raise ValueError("original_index must equal the global 0..N-1 sequence")
        total += chunk.point_count
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
# P-SOURCE v1 Windows lexical comparison key (private)
# ---------------------------------------------------------------------------


def _check_path_component(component: str) -> None:
    if not component:
        raise ValueError("path must not contain empty components")
    if component in (".", ".."):
        raise ValueError("path must not contain dot segments")
    if "\x00" in component:
        raise ValueError("path must not contain a NUL character")
    if component != component.rstrip(" ."):
        raise ValueError("path must not contain trailing whitespace or dots")
    if ":" in component:
        raise ValueError("path must not contain a colon outside the drive letter")


def _frv_source_windows_path_comparison_key(value: str) -> str:
    """Return the frozen pure lexical Windows comparison key for *value*.

    Only absolute Windows drive paths and absolute UNC paths are accepted.
    Relative paths, device-namespace paths, dot segments, empty components,
    trailing whitespace/dots, NUL, and extra colons are rejected.  The key
    uppercases the drive letter, case-folds the server/share and every path
    component, unifies separators to ``\\``, and performs no filesystem
    access, resolution, or cwd dependence.
    """

    path_text = os.fspath(value)
    if type(path_text) is not str:
        raise TypeError("path must be a built-in str")
    if not path_text:
        raise ValueError("path must be non-empty")
    if "\x00" in path_text:
        raise ValueError("path must not contain a NUL character")
    normalized = path_text.replace("/", "\\")
    if normalized.startswith("\\\\?\\") or normalized.startswith("\\\\.\\"):
        raise ValueError("device namespace paths are not supported")
    if normalized.startswith("\\\\"):
        rest = normalized[2:]
        parts = rest.split("\\")
        if len(parts) < 2 or parts[0] == "" or parts[1] == "":
            raise ValueError("UNC path must include a server and share")
        for component in parts:
            _check_path_component(component)
        head = parts[0].casefold() + "\\" + parts[1].casefold()
        tail = "\\".join(component.casefold() for component in parts[2:])
        return "\\\\" + (head if not tail else head + "\\" + tail)
    match = re.match(r"^([A-Za-z]):\\(.*)$", normalized)
    if match is None:
        raise ValueError("path must be an absolute Windows drive or UNC path")
    drive = match.group(1).upper()
    rest = match.group(2)
    if not rest:
        return drive + ":\\"
    components = rest.split("\\")
    for component in components:
        _check_path_component(component)
    return drive + ":\\" + "\\".join(component.casefold() for component in components)


# ---------------------------------------------------------------------------
# Full P-SOURCE reference validator
# ---------------------------------------------------------------------------


def validate_frv_source_provider_reference(
    reference: FrvSourceProviderReference,
    *,
    provider_config: FrvSourceProviderConfig,
    expected_configuration_digest: str,
) -> None:
    """Validate the full frozen cross-binding of a P-SOURCE reference.

    On success returns ``None``.  Argument or field type errors raise
    ``TypeError``; contract cross-binding mismatches raise ``ValueError``.
    """

    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(provider_config) is not FrvSourceProviderConfig:
        raise TypeError("provider_config must be a FrvSourceProviderConfig")
    _require_sha(expected_configuration_digest, "expected_configuration_digest")

    if reference.provider_role is not FrvA2ProviderRole.SOURCE_SCAN:
        raise ValueError("provider_role must be SOURCE_SCAN")
    if reference.provider_key != FRV_SOURCE_PROVIDER_KEY:
        raise ValueError("provider_key must be source_scan")

    computed_config_digest = derive_frv_source_provider_config_digest(
        schema_version=provider_config.schema_version,
        source_model=provider_config.source_model,
        coordinate_conversion_policy=provider_config.coordinate_conversion_policy,
        invalid_point_policy=provider_config.invalid_point_policy,
        canonical_content_policy=provider_config.canonical_content_policy,
        pass_through_dimensions=provider_config.pass_through_dimensions,
    )
    if provider_config.provider_config_digest != computed_config_digest:
        raise ValueError("provider_config digest does not match canonical FRVSRCFG1 fields")
    if reference.provider_config_digest != provider_config.provider_config_digest:
        raise ValueError("provider_config_digest must match the provider config")
    if reference.configuration_digest != expected_configuration_digest:
        raise ValueError("configuration_digest must match the expected configuration digest")

    components = reference.artifact_components
    if len(components) != 1:
        raise ValueError("exactly one artifact component is required")
    component = components[0]
    if component.component_kind is not FrvA2ArtifactComponentKind.SOURCE_FILE:
        raise ValueError("artifact component must be a SOURCE_FILE component")
    if component.component_key != "source_file":
        raise ValueError("artifact component key must be source_file")
    if component.sha256 != reference.source_identity.source_file_sha256:
        raise ValueError("artifact component sha256 must match the source identity")

    expected_reference_id = _derive_artifact_reference_id(
        reference.provider_role, reference.provider_key, reference.artifact_components
    )
    if reference.artifact_reference_id != expected_reference_id:
        raise ValueError("artifact_reference_id does not match the canonical artifact components")

    runtime_key = _frv_source_windows_path_comparison_key(
        os.fspath(reference.runtime_access_path)
    )
    resolved_key = _frv_source_windows_path_comparison_key(
        reference.source_identity.resolved_access_path
    )
    if runtime_key != resolved_key:
        raise ValueError("runtime_access_path must match the resolved access path")

    source = reference.source_identity
    if source.relocation_audit.resolved_path_raw != source.resolved_access_path:
        raise ValueError("relocation audit resolved_path_raw must match resolved_access_path")
    validate_source_identity_record_digest(source)

    if reference.interface_version != FRV_SOURCE_PROVIDER_INTERFACE_VERSION:
        raise ValueError("interface_version must match the frozen interface version")
    if reference.schema_version != FRV_SOURCE_PROVIDER_SCHEMA_VERSION:
        raise ValueError("schema_version must match the frozen schema version")
    if reference.algorithm_version != FRV_SOURCE_PROVIDER_ALGORITHM_VERSION:
        raise ValueError("algorithm_version must match the frozen algorithm version")

    if source.canonical_point_content_digest is None:
        if reference.canonical_point_content_spec is not None:
            raise ValueError(
                "canonical_point_content_spec must be null when the source "
                "declares no canonical content"
            )
    else:
        if reference.canonical_point_content_spec is None:
            raise ValueError(
                "canonical_point_content_spec is required when the source "
                "declares canonical content"
            )
        if (
            reference.canonical_point_content_spec.spec_digest
            != source.canonical_point_content_spec_digest
        ):
            raise ValueError("canonical_point_content_spec digest must match the source identity")
        _check_exact_point_content_spec(reference.canonical_point_content_spec)


# ---------------------------------------------------------------------------
# C3-A: P-SOURCE load error and single-use load session protocol
# ---------------------------------------------------------------------------


class FrvSourceLoadError(Exception):
    """P-SOURCE load failure carrying a frozen A2 failure record.

    The failure record is validated by the existing A2 authority and stored
    verbatim.  ``quality_flags`` is validated by the existing closed
    ``FrvQualityFlag`` bitmask authority (any unknown bit is rejected).
    ``str(error)`` is exactly ``failure_record.detail``; no class name, quality
    flags, underlying exception text, or path is appended.  A raw exception may
    only be attached by Python's native ``raise ... from`` chaining
    (``error.__cause__``); C3-A performs no IO failure mapping.
    """

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


class FrvSourceLoadSession(Protocol):
    """Single-use P-SOURCE load session interface (C3-A contracts only).

    C3-A freezes the five-method interface.  ``iter_chunks`` is a single-use
    runtime contract, ``finalize`` succeeds only after a complete natural
    iteration, ``close`` is idempotent, ``__enter__`` returns the session, and
    ``__exit__`` returns ``False``.  No runtime implementation exists yet; the
    future result type is intentionally left as a forward annotation.
    """

    def iter_chunks(self) -> Iterator[FrvSourceChunk]: ...
    def finalize(self) -> FrvSourceProviderResult: ...
    def close(self) -> None: ...
    def __enter__(self) -> FrvSourceLoadSession: ...
    def __exit__(self, exc_type, exc, traceback) -> Literal[False]: ...


# ---------------------------------------------------------------------------
# C3-B/C3-C: success proof, finalized result, and BND1 evidence adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvSourceProviderProof:
    """Frozen, serializable exact-schema success proof (C3-B).

    A proof records a completed P-SOURCE success state.  Construction is
    structural only: cross-binding checks (frozen role/key/versions, digest
    recomputation, identity/header/domain binding) are enforced by the
    finalized-result validator, not by ``__post_init__``.
    """

    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    configuration_digest: str
    provider_config_digest: str
    artifact_reference_id: str
    artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    source_content_identity_digest: str
    source_identity_record_digest: str
    source_point_count: int
    source_bounds_min_xyz: tuple[float, float, float]
    source_bounds_max_xyz: tuple[float, float, float]
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    domain_kind: FrvA2DomainKind
    domain_count: int
    original_index_domain_digest: str
    sample_index_domain_digest: None
    canonical_point_content_digest: str | None
    canonical_point_content_spec_digest: str | None
    header: FrvSourceHeaderMetadata
    pass_through_dimensions: tuple[FrvSourcePassThroughDimension, ...]

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        _require_exact_str(self.provider_key, "provider_key")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        _require_exact_str(self.interface_version, "interface_version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        _require_sha(self.configuration_digest, "configuration_digest")
        _require_sha(self.provider_config_digest, "provider_config_digest")
        _require_artifact_reference_id(self.artifact_reference_id, "artifact_reference_id")
        components = _require_tuple(self.artifact_components, "artifact_components")
        for index, component in enumerate(components):
            if type(component) is not FrvA2ArtifactComponentBinding:
                raise ValueError(
                    f"artifact_components[{index}] must be a FrvA2ArtifactComponentBinding"
                )
        _require_sha(self.source_content_identity_digest, "source_content_identity_digest")
        _require_sha(self.source_identity_record_digest, "source_identity_record_digest")
        object.__setattr__(
            self,
            "source_point_count",
            _require_positive_int(self.source_point_count, "source_point_count"),
        )
        object.__setattr__(
            self,
            "source_bounds_min_xyz",
            _validate_vec3(self.source_bounds_min_xyz, "source_bounds_min_xyz"),
        )
        object.__setattr__(
            self,
            "source_bounds_max_xyz",
            _validate_vec3(self.source_bounds_max_xyz, "source_bounds_max_xyz"),
        )
        _require_exact_str(self.coordinate_frame, "coordinate_frame")
        _require_exact_str(self.units, "units")
        object.__setattr__(self, "origin_xyz", _validate_vec3(self.origin_xyz, "origin_xyz"))
        _require_exact_str(self.transform_declaration, "transform_declaration")
        _require_enum(self.domain_kind, FrvA2DomainKind, "domain_kind")
        object.__setattr__(
            self, "domain_count", _require_positive_int(self.domain_count, "domain_count")
        )
        _require_sha(self.original_index_domain_digest, "original_index_domain_digest")
        if self.sample_index_domain_digest is not None:
            raise ValueError("sample_index_domain_digest must be None")
        if (self.canonical_point_content_digest is None) != (
            self.canonical_point_content_spec_digest is None
        ):
            raise ValueError(
                "canonical point content digest and spec digest must both be set or both be null"
            )
        if self.canonical_point_content_digest is not None:
            _require_sha(self.canonical_point_content_digest, "canonical_point_content_digest")
            _require_sha(
                self.canonical_point_content_spec_digest,
                "canonical_point_content_spec_digest",
            )
        if type(self.header) is not FrvSourceHeaderMetadata:
            raise ValueError("header must be a FrvSourceHeaderMetadata")
        object.__setattr__(
            self,
            "pass_through_dimensions",
            _validate_enum_tuple(
                self.pass_through_dimensions,
                FrvSourcePassThroughDimension,
                "pass_through_dimensions",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_role": self.provider_role.value,
            "interface_version": self.interface_version,
            "algorithm_version": self.algorithm_version,
            "configuration_digest": self.configuration_digest,
            "provider_config_digest": self.provider_config_digest,
            "artifact_reference_id": self.artifact_reference_id,
            "artifact_components": [
                component.to_dict() for component in self.artifact_components
            ],
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "source_point_count": self.source_point_count,
            "source_bounds_min_xyz": list(self.source_bounds_min_xyz),
            "source_bounds_max_xyz": list(self.source_bounds_max_xyz),
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_xyz": list(self.origin_xyz),
            "transform_declaration": self.transform_declaration,
            "domain_kind": self.domain_kind.value,
            "domain_count": self.domain_count,
            "original_index_domain_digest": self.original_index_domain_digest,
            "sample_index_domain_digest": self.sample_index_domain_digest,
            "canonical_point_content_digest": self.canonical_point_content_digest,
            "canonical_point_content_spec_digest": self.canonical_point_content_spec_digest,
            "header": self.header.to_dict(),
            "pass_through_dimensions": [
                dimension.value for dimension in self.pass_through_dimensions
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvSourceProviderProof:
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
        for name in ("source_bounds_min_xyz", "source_bounds_max_xyz", "origin_xyz"):
            values[name] = _tuple_from_json(values[name], name)
        values["header"] = FrvSourceHeaderMetadata.from_dict(values["header"])
        values["pass_through_dimensions"] = tuple(
            _enum_from_value(
                item, FrvSourcePassThroughDimension, "pass_through_dimensions[]"
            )
            for item in _tuple_from_json(
                values["pass_through_dimensions"], "pass_through_dimensions"
            )
        )
        return cls(**values)


class _FRV_SOURCE_FINALIZED_CAPABILITY:
    """Module-private identity-only finalized P-SOURCE capability token.

    The class object itself is the singleton.  It is never exported,
    serialized, hashed, or compared by value; identity is the only check.
    """

    __slots__ = ()


_MISSING = object()


@dataclass(frozen=True, slots=True, init=False)
class FrvSourceProviderResult:
    """Non-serializable finalized success wrapper (C3-B).

    The only public read-only attributes are ``proof`` and ``source_identity``.
    There is no public constructor: instances are created exclusively by the
    module-private factory after full cross-binding validation.  The private
    capability slot is excluded from repr, equality, serialization, digest,
    proof, and BND1.
    """

    proof: FrvSourceProviderProof
    source_identity: SourceIdentity
    _finalized_capability: Any = field(init=False, repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrvSourceProviderResult has no public constructor")


def _create_finalized_frv_source_provider_result(
    proof: FrvSourceProviderProof,
    source_identity: SourceIdentity,
    *,
    capability,
) -> FrvSourceProviderResult:
    """Private factory for a finalized P-SOURCE result (single authority).

    The concrete session ``finalize()`` is the only authorized production
    caller.  Construction requires the exact module-private capability identity
    and passes the full proof/source cross-binding validation.
    """

    if type(proof) is not FrvSourceProviderProof:
        raise TypeError("proof must be a FrvSourceProviderProof")
    if type(source_identity) is not SourceIdentity:
        raise TypeError("source_identity must be a SourceIdentity")
    if capability is not _FRV_SOURCE_FINALIZED_CAPABILITY:
        raise ValueError("capability must be the module-private finalized capability")
    _validate_finalized_bindings(proof, source_identity)
    result = object.__new__(FrvSourceProviderResult)
    object.__setattr__(result, "proof", proof)
    object.__setattr__(result, "source_identity", source_identity)
    object.__setattr__(result, "_finalized_capability", capability)
    return result


def _validate_finalized_bindings(
    proof: FrvSourceProviderProof, source_identity: SourceIdentity
) -> None:
    """Full cross-binding checks shared by the factory and public validator."""

    if proof.provider_role is not FrvA2ProviderRole.SOURCE_SCAN:
        raise ValueError("provider_role must be SOURCE_SCAN")
    if proof.provider_key != FRV_SOURCE_PROVIDER_KEY:
        raise ValueError("provider_key must be source_scan")
    if (
        proof.interface_version != FRV_SOURCE_PROVIDER_INTERFACE_VERSION
        or proof.schema_version != FRV_SOURCE_PROVIDER_SCHEMA_VERSION
        or proof.algorithm_version != FRV_SOURCE_PROVIDER_ALGORITHM_VERSION
    ):
        raise ValueError("proof versions must match the frozen P-SOURCE versions")

    computed_config_digest = derive_frv_source_provider_config_digest(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=proof.pass_through_dimensions,
    )
    if proof.provider_config_digest != computed_config_digest:
        raise ValueError("provider_config_digest does not match canonical FRVSRCFG1 fields")

    components = proof.artifact_components
    if len(components) != 1:
        raise ValueError("exactly one artifact component is required")
    component = components[0]
    if component.component_kind is not FrvA2ArtifactComponentKind.SOURCE_FILE:
        raise ValueError("artifact component must be a SOURCE_FILE component")
    if component.component_key != "source_file":
        raise ValueError("artifact component key must be source_file")
    if component.sha256 != source_identity.source_file_sha256:
        raise ValueError("artifact component sha256 must match the source identity")
    expected_reference_id = _derive_artifact_reference_id(
        proof.provider_role, proof.provider_key, proof.artifact_components
    )
    if proof.artifact_reference_id != expected_reference_id:
        raise ValueError("artifact_reference_id does not match the canonical artifact components")

    if proof.source_content_identity_digest != source_identity.source_content_identity_digest:
        raise ValueError("source_content_identity_digest must match the source identity")
    if proof.source_identity_record_digest != source_identity.source_identity_record_digest:
        raise ValueError("source_identity_record_digest must match the source identity")

    if proof.source_point_count != source_identity.source_point_count:
        raise ValueError("source_point_count must match the source identity")
    if proof.source_point_count != proof.header.point_count:
        raise ValueError("source_point_count must match the header point count")
    if proof.domain_count != proof.source_point_count:
        raise ValueError("domain_count must equal the source point count")
    if proof.source_point_count <= 0:
        raise ValueError("source point count must be positive")

    if proof.source_bounds_min_xyz != tuple(source_identity.source_bounds_min_xyz):
        raise ValueError("source_bounds_min_xyz must match the source identity")
    if proof.source_bounds_max_xyz != tuple(source_identity.source_bounds_max_xyz):
        raise ValueError("source_bounds_max_xyz must match the source identity")
    if proof.source_bounds_min_xyz != tuple(proof.header.min_xyz):
        raise ValueError("source_bounds_min_xyz must match the header")
    if proof.source_bounds_max_xyz != tuple(proof.header.max_xyz):
        raise ValueError("source_bounds_max_xyz must match the header")

    if proof.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
        raise ValueError("coordinate_frame must be LOCAL_CARTESIAN_XYZ")
    if proof.coordinate_frame != source_identity.coordinate_frame:
        raise ValueError("coordinate_frame must match the source identity")
    if proof.units != "m":
        raise ValueError("units must be m")
    if proof.units != source_identity.units:
        raise ValueError("units must match the source identity")
    if proof.origin_xyz != tuple(source_identity.origin_context):
        raise ValueError("origin_xyz must match the source identity origin context")

    if (proof.canonical_point_content_digest is None) != (
        proof.canonical_point_content_spec_digest is None
    ):
        raise ValueError("canonical point content digest pair must be consistent")
    if proof.canonical_point_content_digest != source_identity.canonical_point_content_digest:
        raise ValueError("canonical_point_content_digest must match the source identity")
    if proof.canonical_point_content_spec_digest != (
        source_identity.canonical_point_content_spec_digest
    ):
        raise ValueError("canonical_point_content_spec_digest must match the source identity")

    if proof.domain_kind is not FrvA2DomainKind.FULL_SOURCE:
        raise ValueError("domain_kind must be FULL_SOURCE")
    expected_index_digest = frv_a2_index_domain_digest(
        "ORIGINAL_INDEX",
        proof.source_content_identity_digest,
        FrvA2DomainKind.FULL_SOURCE,
        tuple(range(proof.domain_count)),
    )
    if proof.original_index_domain_digest != expected_index_digest:
        raise ValueError("original_index_domain_digest does not match the full 0..N-1 domain")
    if proof.sample_index_domain_digest is not None:
        raise ValueError("sample_index_domain_digest must be None")
    if proof.transform_declaration != "identity":
        raise ValueError("transform_declaration must be identity")

    validate_source_identity_record_digest(source_identity)
    validate_source_content_identity_digest(source_identity)


def validate_finalized_frv_source_provider_result(
    result: FrvSourceProviderResult,
) -> None:
    """Validate the frozen finalized-result cross-bindings; return ``None``."""

    if type(result) is not FrvSourceProviderResult:
        raise TypeError("result must be a FrvSourceProviderResult")
    if getattr(result, "_finalized_capability", _MISSING) is not (
        _FRV_SOURCE_FINALIZED_CAPABILITY
    ):
        raise ValueError("result must carry the module-private finalized capability")
    proof = getattr(result, "proof", _MISSING)
    if type(proof) is not FrvSourceProviderProof:
        raise ValueError("result.proof must be a FrvSourceProviderProof")
    source_identity = getattr(result, "source_identity", _MISSING)
    if type(source_identity) is not SourceIdentity:
        raise ValueError("result.source_identity must be a SourceIdentity")
    _validate_finalized_bindings(proof, source_identity)


def to_frv_a2_provider_binding_evidence(
    result: FrvSourceProviderResult,
) -> FrvA2ProviderBindingEvidence:
    """Return the success-only ``FRVA2BND1`` formal provider evidence.

    The finalized-result validator runs before any field is read; only a
    successfully finalized result can reach the mapping.  ``binding_digest`` is
    recomputed by the existing ``frv_a2_provider_binding_digest`` authority.
    """

    validate_finalized_frv_source_provider_result(result)
    proof = result.proof
    evidence = FrvA2ProviderBindingEvidence.__new__(FrvA2ProviderBindingEvidence)
    object.__setattr__(evidence, "provider_key", "source_scan")
    object.__setattr__(evidence, "provider_role", FrvA2ProviderRole.SOURCE_SCAN)
    object.__setattr__(evidence, "evidence_class", FrvA2EvidenceClass.FORMAL_PROVIDER)
    object.__setattr__(evidence, "public_module", "algorithms.io.source_provider")
    object.__setattr__(evidence, "public_type_name", "FrvSourceProviderResult")
    object.__setattr__(evidence, "public_loader_name", "load_frv_source")
    object.__setattr__(
        evidence, "interface_version", FRV_SOURCE_PROVIDER_INTERFACE_VERSION
    )
    object.__setattr__(evidence, "schema_version", FRV_SOURCE_PROVIDER_SCHEMA_VERSION)
    object.__setattr__(evidence, "algorithm_version", FRV_SOURCE_PROVIDER_ALGORITHM_VERSION)
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
    object.__setattr__(evidence, "upstream_binding_digests", ())
    object.__setattr__(evidence, "review_codes", ())
    object.__setattr__(evidence, "limitation_codes", ())
    object.__setattr__(evidence, "binding_digest", frv_a2_provider_binding_digest(evidence))
    return evidence


__all__ = [
    "FRV_SOURCE_PROVIDER_KEY",
    "FRV_SOURCE_PROVIDER_INTERFACE_VERSION",
    "FRV_SOURCE_PROVIDER_SCHEMA_VERSION",
    "FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION",
    "FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION",
    "FRV_SOURCE_PROVIDER_ALGORITHM_VERSION",
    "FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN",
    "FRV_SOURCE_DEFAULT_CHUNK_SIZE",
    "derive_frv_source_provider_config_digest",
    "derive_frv_source_point_content_digest",
    "derive_frv_source_original_index_digest",
    "validate_frv_source_provider_reference",
    "FrvSourceModel",
    "FrvSourceCoordinateConversionPolicy",
    "FrvSourceInvalidPointPolicy",
    "FrvSourceCanonicalContentPolicy",
    "FrvSourceFormat",
    "FrvSourceCompressionState",
    "FrvSourceCrsAuthorityState",
    "FrvSourceVlrPolicyState",
    "FrvSourcePassThroughDimension",
    "FrvSourceProviderConfig",
    "FrvSourceHeaderMetadata",
    "FrvSourceProviderReference",
    "FrvSourceChunk",
    "FrvSourceLoadError",
    "FrvSourceLoadSession",
    "FrvSourceProviderProof",
    "FrvSourceProviderResult",
    "validate_finalized_frv_source_provider_result",
    "to_frv_a2_provider_binding_evidence",
]
