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

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from core.contracts.frv import FrvCanonicalPointContentSpec, SourceIdentity
from core.contracts.frv_a2 import FrvA2ArtifactComponentBinding, FrvA2ProviderRole

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


__all__ = [
    "FRV_SOURCE_PROVIDER_KEY",
    "FRV_SOURCE_PROVIDER_INTERFACE_VERSION",
    "FRV_SOURCE_PROVIDER_SCHEMA_VERSION",
    "FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION",
    "FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION",
    "FRV_SOURCE_PROVIDER_ALGORITHM_VERSION",
    "FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN",
    "FRV_SOURCE_DEFAULT_CHUNK_SIZE",
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
]
