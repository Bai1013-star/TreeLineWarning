"""FRV-A2 P-GEOMETRY provider contracts (PLAN-FROZEN).

P-GEOMETRY consumes a finalized P-SOURCE result and emits full-resolution
multiscale geometry evidence over the complete FULL_SOURCE domain.  This module
freezes the provider config, runtime chunk, success proof, finalized result,
load error, session protocol, validators, digest framings, and the FRVA2BND1
adapter.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from types import TracebackType
from typing import Any, Literal, Protocol

import numpy as np

from core.contracts.frv import (
    FrvErrorCode,
    FrvQualityFlag,
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

# ---------------------------------------------------------------------------
# Public constants (frozen names and values)
# ---------------------------------------------------------------------------

FRV_GEOMETRY_PROVIDER_KEY = "geometry_provider"
FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION = "frv-geometry-provider-v1"
FRV_GEOMETRY_PROVIDER_SCHEMA_VERSION = "frv-geometry-provider-result-v1"
FRV_GEOMETRY_CONFIG_SCHEMA_VERSION = "frv-geometry-provider-config-v1"
FRV_GEOMETRY_PROOF_SCHEMA_VERSION = "frv-geometry-provider-proof-v1"
FRV_GEOMETRY_ALGORITHM_VERSION = "frv-geometry-multiscale-pca-grid-v1"
FRV_GEOMETRY_CONFIG_DIGEST_DOMAIN = "FRVGEOCFG1"
FRV_GEOMETRY_POINT_CONTENT_DIGEST_DOMAIN = "FRVGEOPCD1"
FRV_GEOMETRY_PROOF_DIGEST_DOMAIN = "FRVGEOPRF1"
FRV_GEOMETRY_CACHE_BYTES = 536870912

# Digest domain that binds the exact serialization of the P-SOURCE proof into
# the P-GEOMETRY proof.  The frozen plan does not name a domain for this
# binding; it uses the same framing shape as the P-HAG source-proof binding so
# that the field is always a deterministic lowercase SHA-256.
_FRV_GEOMETRY_SOURCE_PROOF_DIGEST_DOMAIN = "FRVGEOSRC1"

_FRV_GEOMETRY_ARTIFACT_REFERENCE_PREFIX = "frv-a2-ref-v1|"

# ---------------------------------------------------------------------------
# Frozen evidence enums
# ---------------------------------------------------------------------------


class FrvGeometryEvidenceFlag(IntFlag):
    """Closed per-point geometry evidence flags."""

    NONE = 0
    NEIGHBORHOOD_SUFFICIENT = 1
    DENSITY_SUPPORTED = 2
    CANOPY_BRANCHING_COMPATIBLE = 4
    WOODY_COLUMN_COMPATIBLE = 8
    GROUND_SURFACE_COMPATIBLE = 16
    VERTICAL_CONTINUITY = 32


class FrvGeometryEvidenceState(IntEnum):
    """Closed per-point multiscale evidence state."""

    VALID = 1
    INSUFFICIENT = 2
    DEGENERATE = 3


class FrvGeometryReasonFlag(IntFlag):
    """Closed per-point reason flags for non-VALID evidence."""

    NONE = 0
    INSUFFICIENT_NEIGHBORS = 1
    DEGENERATE_COVARIANCE = 2


class FrvGeometryQualityFlag(IntFlag):
    """Closed per-point deterministic truncation quality flags."""

    NONE = 0
    NEIGHBOR_CAP_REACHED = 1
    SUPPORT_NEIGHBOR_CAP_REACHED = 2


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


def _require_unit_float(value: Any, name: str) -> float:
    number = _require_finite_float(value, name)
    if not (0.0 <= number <= 1.0):
        raise ValueError(f"{name} must lie in [0,1]")
    return number


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
    if not text.startswith(_FRV_GEOMETRY_ARTIFACT_REFERENCE_PREFIX):
        raise ValueError(f"{name} must start with {_FRV_GEOMETRY_ARTIFACT_REFERENCE_PREFIX!r}")
    return text


def _normalize_zero(value: float) -> float:
    """Return +0.0 for any exact-zero float, otherwise the finite value."""
    return 0.0 if value == 0.0 else float(value)


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _require_ndarray(
    value: Any,
    name: str,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    finite: bool,
    zero_normalized: bool,
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
    if value.dtype.hasobject:
        raise ValueError(f"{name} must not contain object values")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    if zero_normalized:
        for item in value.ravel():
            if _is_negative_zero(float(item)):
                raise ValueError(f"{name} must not contain IEEE negative zero")


def _stream_index_domain_digest(total: int, source_content_identity_digest: str) -> str:
    """Return the FRVA2IDX1 digest over the global 0..N-1 domain, in-stream."""

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
# FRVGEOCFG1 provider-config digest
# ---------------------------------------------------------------------------


def derive_frv_geometry_provider_config_digest(
    *,
    schema_version: str,
    provider_key: str,
    provider_role: FrvA2ProviderRole,
    interface_version: str,
    algorithm_version: str,
    radii_m: tuple[float, float, float],
    min_neighbor_counts: tuple[int, int, int],
    max_neighbors_per_scale: int,
    index_cell_size_m: float,
    support_neighbor_radius_m: float,
    support_neighbor_cap: int,
    direction_cosine_min: float,
    eigenvalue_floor_m2: float,
    canopy_scattering_min: float,
    canopy_linearity_max: float,
    canopy_density_min_per_m3: float,
    woody_linearity_min: float,
    woody_verticality_max: float,
    woody_vertical_span_min_m: float,
    vertical_cylinder_radius_m: float,
    vertical_half_window_m: float,
    vertical_slice_height_m: float,
    min_vertical_occupied_slices: int,
    ground_planarity_min: float,
    ground_normal_z_min: float,
    max_index_cell_points: int,
) -> str:
    """Return the FRVGEOCFG1 digest for the frozen provider-config payload."""

    payload = {
        "digest_domain": FRV_GEOMETRY_CONFIG_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "provider_key": provider_key,
        "provider_role": provider_role.value,
        "interface_version": interface_version,
        "algorithm_version": algorithm_version,
        "radii_m": list(radii_m),
        "min_neighbor_counts": list(min_neighbor_counts),
        "max_neighbors_per_scale": max_neighbors_per_scale,
        "index_cell_size_m": index_cell_size_m,
        "support_neighbor_radius_m": support_neighbor_radius_m,
        "support_neighbor_cap": support_neighbor_cap,
        "direction_cosine_min": direction_cosine_min,
        "eigenvalue_floor_m2": eigenvalue_floor_m2,
        "canopy_scattering_min": canopy_scattering_min,
        "canopy_linearity_max": canopy_linearity_max,
        "canopy_density_min_per_m3": canopy_density_min_per_m3,
        "woody_linearity_min": woody_linearity_min,
        "woody_verticality_max": woody_verticality_max,
        "woody_vertical_span_min_m": woody_vertical_span_min_m,
        "vertical_cylinder_radius_m": vertical_cylinder_radius_m,
        "vertical_half_window_m": vertical_half_window_m,
        "vertical_slice_height_m": vertical_slice_height_m,
        "min_vertical_occupied_slices": min_vertical_occupied_slices,
        "ground_planarity_min": ground_planarity_min,
        "ground_normal_z_min": ground_normal_z_min,
        "max_index_cell_points": max_index_cell_points,
    }
    return frv_digest(payload)


# ---------------------------------------------------------------------------
# FrvGeometryProviderConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvGeometryProviderConfig:
    """Exact-schema P-GEOMETRY provider configuration (domain FRVGEOCFG1)."""

    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    radii_m: tuple[float, float, float]
    min_neighbor_counts: tuple[int, int, int]
    max_neighbors_per_scale: int
    index_cell_size_m: float
    support_neighbor_radius_m: float
    support_neighbor_cap: int
    direction_cosine_min: float
    eigenvalue_floor_m2: float
    canopy_scattering_min: float
    canopy_linearity_max: float
    canopy_density_min_per_m3: float
    woody_linearity_min: float
    woody_verticality_max: float
    woody_vertical_span_min_m: float
    vertical_cylinder_radius_m: float
    vertical_half_window_m: float
    vertical_slice_height_m: float
    min_vertical_occupied_slices: int
    ground_planarity_min: float
    ground_normal_z_min: float
    max_index_cell_points: int
    provider_config_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_GEOMETRY_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported P-GEOMETRY provider config schema version")
        _require_exact_str(self.provider_key, "provider_key")
        if self.provider_key != FRV_GEOMETRY_PROVIDER_KEY:
            raise ValueError("provider_key must be geometry_provider")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        if self.provider_role is not FrvA2ProviderRole.GEOMETRY:
            raise ValueError("provider_role must be GEOMETRY")
        _require_exact_str(self.interface_version, "interface_version")
        if self.interface_version != FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION:
            raise ValueError("interface_version must match the frozen interface version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != FRV_GEOMETRY_ALGORITHM_VERSION:
            raise ValueError("algorithm_version must match the frozen algorithm version")

        radii = _validate_vec3(self.radii_m, "radii_m")
        object.__setattr__(self, "radii_m", radii)
        for index in range(2):
            if radii[index] <= 0.0 or not (radii[index] < radii[index + 1]):
                raise ValueError("radii_m must be strictly increasing positive values")

        min_counts = self.min_neighbor_counts
        object.__setattr__(
            self, "min_neighbor_counts", _require_tuple(min_counts, "min_neighbor_counts")
        )
        if len(min_counts) != 3:
            raise ValueError("min_neighbor_counts must contain exactly three values")
        for index, count in enumerate(min_counts):
            if type(count) is not int:
                raise ValueError(f"min_neighbor_counts[{index}] must be a built-in int")
            if count <= 0 or count > 64:
                raise ValueError(
                    f"min_neighbor_counts[{index}] must be positive and at most 64"
                )

        object.__setattr__(
            self,
            "max_neighbors_per_scale",
            _require_positive_int(self.max_neighbors_per_scale, "max_neighbors_per_scale"),
        )
        if self.max_neighbors_per_scale < max(min_counts):
            raise ValueError("max_neighbors_per_scale must be >= every minimum count")

        object.__setattr__(
            self,
            "index_cell_size_m",
            _require_finite_float(self.index_cell_size_m, "index_cell_size_m"),
        )
        if self.index_cell_size_m <= 0.0:
            raise ValueError("index_cell_size_m must be positive")

        object.__setattr__(
            self,
            "support_neighbor_radius_m",
            _require_finite_float(
                self.support_neighbor_radius_m, "support_neighbor_radius_m"
            ),
        )
        if self.support_neighbor_radius_m != radii[0]:
            raise ValueError("support_neighbor_radius_m must equal radii_m[0]")
        object.__setattr__(
            self,
            "support_neighbor_cap",
            _require_positive_int(self.support_neighbor_cap, "support_neighbor_cap"),
        )
        object.__setattr__(
            self,
            "direction_cosine_min",
            _require_unit_float(self.direction_cosine_min, "direction_cosine_min"),
        )
        object.__setattr__(
            self,
            "eigenvalue_floor_m2",
            _require_finite_float(self.eigenvalue_floor_m2, "eigenvalue_floor_m2"),
        )
        if self.eigenvalue_floor_m2 <= 0.0:
            raise ValueError("eigenvalue_floor_m2 must be positive")
        object.__setattr__(
            self,
            "canopy_scattering_min",
            _require_unit_float(self.canopy_scattering_min, "canopy_scattering_min"),
        )
        object.__setattr__(
            self,
            "canopy_linearity_max",
            _require_unit_float(self.canopy_linearity_max, "canopy_linearity_max"),
        )
        object.__setattr__(
            self,
            "canopy_density_min_per_m3",
            _require_finite_float(
                self.canopy_density_min_per_m3, "canopy_density_min_per_m3"
            ),
        )
        if self.canopy_density_min_per_m3 <= 0.0:
            raise ValueError("canopy_density_min_per_m3 must be positive")
        object.__setattr__(
            self,
            "woody_linearity_min",
            _require_unit_float(self.woody_linearity_min, "woody_linearity_min"),
        )
        object.__setattr__(
            self,
            "woody_verticality_max",
            _require_unit_float(self.woody_verticality_max, "woody_verticality_max"),
        )
        object.__setattr__(
            self,
            "woody_vertical_span_min_m",
            _require_finite_float(self.woody_vertical_span_min_m, "woody_vertical_span_min_m"),
        )
        if self.woody_vertical_span_min_m <= 0.0:
            raise ValueError("woody_vertical_span_min_m must be positive")
        object.__setattr__(
            self,
            "vertical_cylinder_radius_m",
            _require_finite_float(self.vertical_cylinder_radius_m, "vertical_cylinder_radius_m"),
        )
        if self.vertical_cylinder_radius_m <= 0.0:
            raise ValueError("vertical_cylinder_radius_m must be positive")
        object.__setattr__(
            self,
            "vertical_half_window_m",
            _require_finite_float(self.vertical_half_window_m, "vertical_half_window_m"),
        )
        if self.vertical_half_window_m != radii[2]:
            raise ValueError("vertical_half_window_m must equal radii_m[2]")
        object.__setattr__(
            self,
            "vertical_slice_height_m",
            _require_finite_float(self.vertical_slice_height_m, "vertical_slice_height_m"),
        )
        if self.vertical_slice_height_m <= 0.0:
            raise ValueError("vertical_slice_height_m must be positive")
        if not (2 * self.vertical_half_window_m / self.vertical_slice_height_m == 12):
            raise ValueError("2 * vertical_half_window_m / vertical_slice_height_m must be 12")
        object.__setattr__(
            self,
            "min_vertical_occupied_slices",
            _require_positive_int(
                self.min_vertical_occupied_slices, "min_vertical_occupied_slices"
            ),
        )
        object.__setattr__(
            self,
            "ground_planarity_min",
            _require_unit_float(self.ground_planarity_min, "ground_planarity_min"),
        )
        object.__setattr__(
            self,
            "ground_normal_z_min",
            _require_unit_float(self.ground_normal_z_min, "ground_normal_z_min"),
        )
        object.__setattr__(
            self,
            "max_index_cell_points",
            _require_positive_int(self.max_index_cell_points, "max_index_cell_points"),
        )
        _require_sha(self.provider_config_digest, "provider_config_digest")
        computed = derive_frv_geometry_provider_config_digest(
            schema_version=self.schema_version,
            provider_key=self.provider_key,
            provider_role=self.provider_role,
            interface_version=self.interface_version,
            algorithm_version=self.algorithm_version,
            radii_m=self.radii_m,
            min_neighbor_counts=self.min_neighbor_counts,
            max_neighbors_per_scale=self.max_neighbors_per_scale,
            index_cell_size_m=self.index_cell_size_m,
            support_neighbor_radius_m=self.support_neighbor_radius_m,
            support_neighbor_cap=self.support_neighbor_cap,
            direction_cosine_min=self.direction_cosine_min,
            eigenvalue_floor_m2=self.eigenvalue_floor_m2,
            canopy_scattering_min=self.canopy_scattering_min,
            canopy_linearity_max=self.canopy_linearity_max,
            canopy_density_min_per_m3=self.canopy_density_min_per_m3,
            woody_linearity_min=self.woody_linearity_min,
            woody_verticality_max=self.woody_verticality_max,
            woody_vertical_span_min_m=self.woody_vertical_span_min_m,
            vertical_cylinder_radius_m=self.vertical_cylinder_radius_m,
            vertical_half_window_m=self.vertical_half_window_m,
            vertical_slice_height_m=self.vertical_slice_height_m,
            min_vertical_occupied_slices=self.min_vertical_occupied_slices,
            ground_planarity_min=self.ground_planarity_min,
            ground_normal_z_min=self.ground_normal_z_min,
            max_index_cell_points=self.max_index_cell_points,
        )
        if self.provider_config_digest != computed:
            raise ValueError(
                "provider_config_digest does not match canonical FRVGEOCFG1 fields"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_role": self.provider_role.value,
            "interface_version": self.interface_version,
            "algorithm_version": self.algorithm_version,
            "radii_m": list(self.radii_m),
            "min_neighbor_counts": list(self.min_neighbor_counts),
            "max_neighbors_per_scale": self.max_neighbors_per_scale,
            "index_cell_size_m": self.index_cell_size_m,
            "support_neighbor_radius_m": self.support_neighbor_radius_m,
            "support_neighbor_cap": self.support_neighbor_cap,
            "direction_cosine_min": self.direction_cosine_min,
            "eigenvalue_floor_m2": self.eigenvalue_floor_m2,
            "canopy_scattering_min": self.canopy_scattering_min,
            "canopy_linearity_max": self.canopy_linearity_max,
            "canopy_density_min_per_m3": self.canopy_density_min_per_m3,
            "woody_linearity_min": self.woody_linearity_min,
            "woody_verticality_max": self.woody_verticality_max,
            "woody_vertical_span_min_m": self.woody_vertical_span_min_m,
            "vertical_cylinder_radius_m": self.vertical_cylinder_radius_m,
            "vertical_half_window_m": self.vertical_half_window_m,
            "vertical_slice_height_m": self.vertical_slice_height_m,
            "min_vertical_occupied_slices": self.min_vertical_occupied_slices,
            "ground_planarity_min": self.ground_planarity_min,
            "ground_normal_z_min": self.ground_normal_z_min,
            "max_index_cell_points": self.max_index_cell_points,
            "provider_config_digest": self.provider_config_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvGeometryProviderConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_role"] = _enum_from_value(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        values["radii_m"] = _tuple_from_json(values["radii_m"], "radii_m")
        values["min_neighbor_counts"] = _tuple_from_json(
            values["min_neighbor_counts"], "min_neighbor_counts"
        )
        return cls(**values)


def validate_frv_geometry_provider_config(config: FrvGeometryProviderConfig) -> None:
    if type(config) is not FrvGeometryProviderConfig:
        raise ValueError("config must be a FrvGeometryProviderConfig")
    computed = derive_frv_geometry_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        radii_m=config.radii_m,
        min_neighbor_counts=config.min_neighbor_counts,
        max_neighbors_per_scale=config.max_neighbors_per_scale,
        index_cell_size_m=config.index_cell_size_m,
        support_neighbor_radius_m=config.support_neighbor_radius_m,
        support_neighbor_cap=config.support_neighbor_cap,
        direction_cosine_min=config.direction_cosine_min,
        eigenvalue_floor_m2=config.eigenvalue_floor_m2,
        canopy_scattering_min=config.canopy_scattering_min,
        canopy_linearity_max=config.canopy_linearity_max,
        canopy_density_min_per_m3=config.canopy_density_min_per_m3,
        woody_linearity_min=config.woody_linearity_min,
        woody_verticality_max=config.woody_verticality_max,
        woody_vertical_span_min_m=config.woody_vertical_span_min_m,
        vertical_cylinder_radius_m=config.vertical_cylinder_radius_m,
        vertical_half_window_m=config.vertical_half_window_m,
        vertical_slice_height_m=config.vertical_slice_height_m,
        min_vertical_occupied_slices=config.min_vertical_occupied_slices,
        ground_planarity_min=config.ground_planarity_min,
        ground_normal_z_min=config.ground_normal_z_min,
        max_index_cell_points=config.max_index_cell_points,
    )
    if config.provider_config_digest != computed:
        raise ValueError("provider_config_digest does not match canonical FRVGEOCFG1 fields")


# ---------------------------------------------------------------------------
# FrvGeometryChunk (runtime-only immutable contract)
# ---------------------------------------------------------------------------

_SCALE_0 = 0
_SCALE_1 = 1
_SCALE_2 = 2

_VALID_STATES = frozenset(int(member) for member in FrvGeometryEvidenceState)


@dataclass(frozen=True, slots=True)
class FrvGeometryChunk:
    """Read-only P-GEOMETRY chunk with a one-to-one source-chunk mapping."""

    source_id: str
    chunk_id: int
    start_index: int
    point_count: int
    original_index: np.ndarray
    raw_neighbor_count: np.ndarray
    neighbor_count: np.ndarray
    density_per_m3: np.ndarray
    local_spacing_m: np.ndarray
    eigenvalues_m2: np.ndarray
    principal_direction: np.ndarray
    surface_normal: np.ndarray
    linearity: np.ndarray
    planarity: np.ndarray
    scattering: np.ndarray
    verticality: np.ndarray
    vertical_span_m: np.ndarray
    vertical_occupied_slices: np.ndarray
    above_count: np.ndarray
    below_count: np.ndarray
    support_neighbor_count: np.ndarray
    support_neighbor_original_index: np.ndarray
    geometry_evidence_flags: np.ndarray
    geometry_evidence_state: np.ndarray
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
        specs = _FRV_GEOMETRY_CHUNK_ARRAY_SPECS
        for name, dtype, shape, finite, zero_normalized in specs:
            resolved = (point_count,) + shape
            _require_ndarray(
                getattr(self, name),
                name,
                dtype=np.dtype(dtype),
                shape=resolved,
                finite=finite,
                zero_normalized=zero_normalized,
            )


def validate_frv_geometry_chunk(
    chunk: FrvGeometryChunk,
    *,
    provider_config: FrvGeometryProviderConfig,
) -> None:
    """Validate the frozen per-row geometry semantics against the provider config."""

    if type(chunk) is not FrvGeometryChunk:
        raise ValueError("chunk must be a FrvGeometryChunk")
    if type(provider_config) is not FrvGeometryProviderConfig:
        raise ValueError("provider_config must be a FrvGeometryProviderConfig")
    config = provider_config
    min_counts = config.min_neighbor_counts
    max_neighbors = config.max_neighbors_per_scale
    floor = config.eigenvalue_floor_m2
    support_cap = config.support_neighbor_cap
    previous: int | None = None
    for row in range(chunk.point_count):
        index = int(chunk.original_index[row])
        if index < 0:
            raise ValueError("original_index must be nonnegative")
        if previous is not None and index <= previous:
            raise ValueError("original_index must be strictly increasing within chunk")
        previous = index
        state = int(chunk.geometry_evidence_state[row])
        if state not in _VALID_STATES:
            raise ValueError("geometry_evidence_state must be a FrvGeometryEvidenceState value")
        flags = int(chunk.geometry_evidence_flags[row])
        if flags < 0 or flags > int(
            FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY
            | FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE
            | FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE
            | FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE
            | FrvGeometryEvidenceFlag.DENSITY_SUPPORTED
            | FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT
        ):
            raise ValueError("geometry_evidence_flags contains an unknown bit")
        reason = int(chunk.reason_flags[row])
        quality = int(chunk.quality_flags[row])
        valid_quality = int(
            FrvGeometryQualityFlag.NEIGHBOR_CAP_REACHED
            | FrvGeometryQualityFlag.SUPPORT_NEIGHBOR_CAP_REACHED
        )
        if quality < 0 or quality > valid_quality or quality & (~valid_quality):
            raise ValueError("quality_flags contains an unknown bit")

        neighbor_counts = [0, 0, 0]
        raw_counts = [0, 0, 0]
        densities = [0.0, 0.0, 0.0]
        spacings = [0.0, 0.0, 0.0]
        lambdas = [None, None, None]
        for scale in range(3):
            raw = int(chunk.raw_neighbor_count[row, scale])
            count = int(chunk.neighbor_count[row, scale])
            if count > raw or count > max_neighbors:
                raise ValueError("neighbor_count must not exceed the raw count or the cap")
            raw_counts[scale] = raw
            neighbor_counts[scale] = count
            radius = float(config.radii_m[scale])
            volume = (4.0 / 3.0) * math.pi * radius**3
            density64 = raw / volume
            densities[scale] = density64
            stored_density = float(chunk.density_per_m3[row, scale])
            if stored_density < 0.0:
                raise ValueError("density_per_m3 must be nonnegative")
            if not math.isclose(stored_density, density64, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("density_per_m3 does not match the raw neighbor count")
            stored_spacing = float(chunk.local_spacing_m[row, scale])
            if stored_spacing < 0.0:
                raise ValueError("local_spacing_m must be nonnegative")
            if count == 0 and stored_spacing != 0.0:
                raise ValueError("local_spacing_m must be +0.0 when no neighbors are taken")
            spacings[scale] = stored_spacing

            eigen = [
                float(chunk.eigenvalues_m2[row, scale, axis]) for axis in range(3)
            ]
            if count < int(min_counts[scale]):
                for axis in range(3):
                    if eigen[axis] != 0.0:
                        raise ValueError("eigenvalues must be +0.0 below the scale minimum")
                if not _row_all_zero(chunk.principal_direction[row, scale]):
                    raise ValueError("principal_direction must be +0.0 below the scale minimum")
                if not _row_all_zero(chunk.surface_normal[row, scale]):
                    raise ValueError("surface_normal must be +0.0 below the scale minimum")
                if not _feature_row_zero(chunk, row, scale):
                    raise ValueError("shape features must be +0.0 below the scale minimum")
                lambdas[scale] = (0.0, 0.0, 0.0)
                continue
            if not (eigen[0] <= eigen[1] <= eigen[2]):
                raise ValueError("eigenvalues must be non-decreasing")
            if eigen[0] < 0.0:
                raise ValueError("eigenvalues must be nonnegative after clamping")
            if eigen[2] <= floor:
                if not _row_all_zero(chunk.principal_direction[row, scale]):
                    raise ValueError(
                        "principal_direction must be +0.0 for degenerate covariance"
                    )
                if not _row_all_zero(chunk.surface_normal[row, scale]):
                    raise ValueError("surface_normal must be +0.0 for degenerate covariance")
                if not _feature_row_zero(chunk, row, scale):
                    raise ValueError("shape features must be +0.0 for degenerate covariance")
                lambdas[scale] = tuple(eigen)
                continue
            linearity = float(chunk.linearity[row, scale])
            planarity = float(chunk.planarity[row, scale])
            scattering = float(chunk.scattering[row, scale])
            if not (0.0 <= linearity <= 1.0):
                raise ValueError("linearity must lie in [0,1]")
            if not (0.0 <= planarity <= 1.0):
                raise ValueError("planarity must lie in [0,1]")
            if not (0.0 <= scattering <= 1.0):
                raise ValueError("scattering must lie in [0,1]")
            if not math.isclose(
                linearity, (eigen[2] - eigen[1]) / eigen[2], rel_tol=1e-4, abs_tol=1e-6
            ):
                raise ValueError("linearity does not match the stored eigenvalues")
            if not math.isclose(
                planarity, (eigen[1] - eigen[0]) / eigen[2], rel_tol=1e-4, abs_tol=1e-6
            ):
                raise ValueError("planarity does not match the stored eigenvalues")
            if not math.isclose(
                scattering, eigen[0] / eigen[2], rel_tol=1e-4, abs_tol=1e-6
            ):
                raise ValueError("scattering does not match the stored eigenvalues")
            direction = chunk.principal_direction[row, scale]
            verticality = float(chunk.verticality[row, scale])
            if eigen[2] - eigen[1] <= floor:
                if not _row_all_zero(direction):
                    raise ValueError("principal_direction must be +0.0 on a repeated gap")
                if verticality != 1.0:
                    raise ValueError("verticality must be 1.0 on a repeated gap")
            else:
                if not math.isclose(
                    verticality, 1.0 - abs(float(direction[2])), rel_tol=1e-4, abs_tol=1e-6
                ):
                    raise ValueError("verticality does not match the principal direction")
                if _row_all_zero(direction):
                    raise ValueError("principal_direction must be nonzero on a positive gap")
            normal = chunk.surface_normal[row, scale]
            normal_z = abs(float(normal[2]))
            if eigen[1] - eigen[0] <= floor:
                if not _row_all_zero(normal):
                    raise ValueError("surface_normal must be +0.0 on a repeated gap")
                if normal_z != 0.0:
                    raise ValueError("surface_normal_z must be 0.0 on a repeated gap")
            else:
                if _row_all_zero(normal):
                    raise ValueError("surface_normal must be nonzero on a positive gap")
            lambdas[scale] = tuple(eigen)

        vertical_span = float(chunk.vertical_span_m[row])
        if vertical_span < 0.0:
            raise ValueError("vertical_span_m must be nonnegative")
        occupied = int(chunk.vertical_occupied_slices[row])
        if occupied < 0 or occupied > 12:
            raise ValueError("vertical_occupied_slices must lie in [0,12]")
        above = int(chunk.above_count[row])
        below = int(chunk.below_count[row])
        if above < 0 or below < 0:
            raise ValueError("above_count and below_count must be nonnegative")

        support_count = int(chunk.support_neighbor_count[row])
        if support_count > support_cap:
            raise ValueError("support_neighbor_count must not exceed the support cap")
        support = chunk.support_neighbor_original_index[row]
        seen_support: set[int] = set()
        for slot in range(support_cap):
            value = int(support[slot])
            if slot < support_count:
                if value < 0 or value == index:
                    raise ValueError("support neighbors must be distinct non-self indices")
                if value in seen_support:
                    raise ValueError("support neighbors must be unique")
                seen_support.add(value)
            elif value != -1:
                raise ValueError("unused support slots must be -1")

        # State and reason flags must be mutually consistent.
        medium_at = neighbor_counts[_SCALE_1] >= int(min_counts[_SCALE_1])
        large_at = neighbor_counts[_SCALE_2] >= int(min_counts[_SCALE_2])
        medium_degenerate = medium_at and lambdas[_SCALE_1][2] <= floor
        large_degenerate = large_at and lambdas[_SCALE_2][2] <= floor
        if not (medium_at and large_at):
            expected_state = int(FrvGeometryEvidenceState.INSUFFICIENT)
            expected_reason = int(FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS)
        elif medium_degenerate or large_degenerate:
            expected_state = int(FrvGeometryEvidenceState.DEGENERATE)
            expected_reason = int(FrvGeometryReasonFlag.DEGENERATE_COVARIANCE)
        else:
            expected_state = int(FrvGeometryEvidenceState.VALID)
            expected_reason = int(FrvGeometryReasonFlag.NONE)
        if state != expected_state:
            raise ValueError("geometry_evidence_state does not match the scale counts")
        if reason != expected_reason:
            raise ValueError("reason_flags do not match the evidence state")

        # Deterministic cap quality flag (the raw counts are exact integers).
        expected_quality = 0
        if any(raw > max_neighbors for raw in raw_counts):
            expected_quality |= int(FrvGeometryQualityFlag.NEIGHBOR_CAP_REACHED)
        if expected_quality != quality & int(
            FrvGeometryQualityFlag.NEIGHBOR_CAP_REACHED
        ):
            raise ValueError("NEIGHBOR_CAP_REACHED does not match the raw neighbor counts")

        # Evidence flags must be the exact recomputed set.
        expected_flags = _recompute_evidence_flags(chunk, row, config, densities, state)
        if flags != expected_flags:
            raise ValueError("geometry_evidence_flags do not match the stored evidence")


def _row_all_zero(row_values: Any) -> bool:
    return bool(np.all(row_values == 0.0))


def _feature_row_zero(chunk: FrvGeometryChunk, row: int, scale: int) -> bool:
    values = (
        chunk.linearity[row, scale],
        chunk.planarity[row, scale],
        chunk.scattering[row, scale],
        chunk.verticality[row, scale],
    )
    return bool(np.all(np.asarray(values) == 0.0))


def _recompute_evidence_flags(
    chunk: FrvGeometryChunk,
    row: int,
    config: FrvGeometryProviderConfig,
    densities: list[float],
    state: int,
) -> int:
    """Recompute the frozen evidence flags from the stored chunk values.

    Threshold comparisons use the stored float64 density (recomputed from the
    exact raw counts) and the stored float32 display features, mirroring the
    provider's float64-before-float32 decision discipline.
    """

    flags = 0
    valid = state == int(FrvGeometryEvidenceState.VALID)
    density_supported = densities[_SCALE_2] >= config.canopy_density_min_per_m3
    vertical_continuity = (
        float(chunk.vertical_span_m[row]) >= config.woody_vertical_span_min_m
        and int(chunk.vertical_occupied_slices[row]) >= config.min_vertical_occupied_slices
        and int(chunk.above_count[row]) >= 1
        and int(chunk.below_count[row]) >= 1
    )
    if valid:
        flags |= int(FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT)
    if density_supported:
        flags |= int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED)
    if valid and density_supported:
        if (
            float(chunk.scattering[row, _SCALE_2]) >= config.canopy_scattering_min
            and float(chunk.linearity[row, _SCALE_2]) <= config.canopy_linearity_max
        ):
            flags |= int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)
    if vertical_continuity:
        flags |= int(FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY)
    if valid and vertical_continuity:
        if (
            float(chunk.linearity[row, _SCALE_1]) >= config.woody_linearity_min
            and float(chunk.verticality[row, _SCALE_1]) <= config.woody_verticality_max
        ):
            flags |= int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE)
    if valid:
        if (
            float(chunk.planarity[row, _SCALE_1]) >= config.ground_planarity_min
            and abs(float(chunk.surface_normal[row, _SCALE_1, 2]))
            >= config.ground_normal_z_min
        ):
            flags |= int(FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE)
    return flags


# ---------------------------------------------------------------------------
# FRVGEOPCD1 geometry point-content digest
# ---------------------------------------------------------------------------

# Exact ordered field framing for FRVGEOPCD1 (field name, dtype, row element
# count).  source_id, chunk scalars, and is_last are excluded.
_FRV_GEOMETRY_PCD_FIELDS = (
    ("original_index", "<i8", 1),
    ("raw_neighbor_count", "<u4", 3),
    ("neighbor_count", "<u2", 3),
    ("density_per_m3", "<f4", 3),
    ("local_spacing_m", "<f4", 3),
    ("eigenvalues_m2", "<f8", 9),
    ("principal_direction", "<f4", 9),
    ("surface_normal", "<f4", 9),
    ("linearity", "<f4", 3),
    ("planarity", "<f4", 3),
    ("scattering", "<f4", 3),
    ("verticality", "<f4", 3),
    ("vertical_span_m", "<f4", 1),
    ("vertical_occupied_slices", "<u2", 1),
    ("above_count", "<u2", 1),
    ("below_count", "<u2", 1),
    ("support_neighbor_count", "|u1", 1),
    ("support_neighbor_original_index", "<i8", 16),
    ("geometry_evidence_flags", "<u4", 1),
    ("geometry_evidence_state", "|u1", 1),
    ("reason_flags", "<u4", 1),
    ("quality_flags", "<u4", 1),
)

_FRV_GEOMETRY_PCD_NAMES = frozenset(name for name, _dtype, _count in _FRV_GEOMETRY_PCD_FIELDS)

_FRV_GEOMETRY_CHUNK_ARRAY_SPECS = (
    ("original_index", "<i8", (), False, False),
    ("raw_neighbor_count", "<u4", (3,), False, False),
    ("neighbor_count", "<u2", (3,), False, False),
    ("density_per_m3", "<f4", (3,), True, True),
    ("local_spacing_m", "<f4", (3,), True, True),
    ("eigenvalues_m2", "<f8", (3, 3), True, True),
    ("principal_direction", "<f4", (3, 3), True, True),
    ("surface_normal", "<f4", (3, 3), True, True),
    ("linearity", "<f4", (3,), True, True),
    ("planarity", "<f4", (3,), True, True),
    ("scattering", "<f4", (3,), True, True),
    ("verticality", "<f4", (3,), True, True),
    ("vertical_span_m", "<f4", (), True, True),
    ("vertical_occupied_slices", "<u2", (), False, False),
    ("above_count", "<u2", (), False, False),
    ("below_count", "<u2", (), False, False),
    ("support_neighbor_count", "|u1", (), False, False),
    ("support_neighbor_original_index", "<i8", (16,), False, False),
    ("geometry_evidence_flags", "<u4", (), False, False),
    ("geometry_evidence_state", "|u1", (), False, False),
    ("reason_flags", "<u4", (), False, False),
    ("quality_flags", "<u4", (), False, False),
)


def frv_geometry_point_content_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    provider_config_digest: str,
    source_point_count: int,
    chunks: Iterable[FrvGeometryChunk],
) -> str:
    """Return the FRVGEOPCD1 digest over the complete 0..N-1 geometry domain.

    The framing is ASCII ``FRVGEOPCD1\\0``, three NUL-terminated SHA-256
    headers (source content, source record, provider config), ``uint64 N``,
    then for every ndarray field in frozen order a ``uint16`` field-name byte
    length, the UTF-8 field name, a ``uint32`` single-row byte length, and the
    little-endian C-order row bytes in original_index order.  Exact zeros are
    normalized to ``+0.0`` before hashing.
    """

    hasher = hashlib.sha256()
    hasher.update(b"FRVGEOPCD1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(provider_config_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", source_point_count))
    expected_index = 0
    for chunk in chunks:
        if type(chunk) is not FrvGeometryChunk:
            raise TypeError("chunks must contain FrvGeometryChunk values")
        for row in range(chunk.point_count):
            index = int(chunk.original_index[row])
            if index != expected_index:
                raise ValueError("chunks must cover the complete 0..N-1 domain")
            expected_index += 1
    if expected_index != source_point_count:
        raise ValueError("chunks must cover the complete 0..N-1 domain")

    for name, dtype_str, count in _FRV_GEOMETRY_PCD_FIELDS:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError("field name too long for the FRVGEOPCD1 framing")
        dtype = np.dtype(dtype_str)
        row_byte_length = dtype.itemsize * count
        if row_byte_length > 0xFFFFFFFF:
            raise ValueError("row byte length too long for the FRVGEOPCD1 framing")
        hasher.update(struct.pack("<H", len(name_bytes)))
        hasher.update(name_bytes)
        hasher.update(struct.pack("<I", row_byte_length))
        for chunk in chunks:
            array = getattr(chunk, name)
            for row in range(chunk.point_count):
                if dtype_str in ("<f4", "<f8"):
                    row_bytes = (array[row] + np.zeros(1, dtype=dtype)).tobytes()
                else:
                    row_bytes = array[row].tobytes()
                hasher.update(row_bytes)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# FrvGeometryProviderProof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvGeometryProviderProof:
    """Frozen, serializable exact-schema P-GEOMETRY success proof."""

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
    geometry_point_content_digest: str
    valid_count: int
    insufficient_count: int
    degenerate_count: int
    neighborhood_sufficient_count: int
    density_supported_count: int
    canopy_branching_compatible_count: int
    woody_column_compatible_count: int
    ground_surface_compatible_count: int
    vertical_continuity_count: int
    neighbor_cap_reached_count: int
    support_neighbor_cap_reached_count: int
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    quality_flags: int
    reason_flags: int
    error_codes: tuple[FrvErrorCode, ...]
    proof_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_GEOMETRY_PROOF_SCHEMA_VERSION:
            raise ValueError("unsupported P-GEOMETRY proof schema version")
        _require_exact_str(self.provider_key, "provider_key")
        if self.provider_key != FRV_GEOMETRY_PROVIDER_KEY:
            raise ValueError("provider_key must be geometry_provider")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        if self.provider_role is not FrvA2ProviderRole.GEOMETRY:
            raise ValueError("provider_role must be GEOMETRY")
        _require_exact_str(self.interface_version, "interface_version")
        if self.interface_version != FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION:
            raise ValueError("interface_version must match the frozen interface version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != FRV_GEOMETRY_ALGORITHM_VERSION:
            raise ValueError("algorithm_version must match the frozen algorithm version")
        _require_sha(self.configuration_digest, "configuration_digest")
        _require_sha(self.provider_config_digest, "provider_config_digest")
        _require_artifact_reference_id(
            self.source_artifact_reference_id, "source_artifact_reference_id"
        )
        _require_artifact_reference_id(self.artifact_reference_id, "artifact_reference_id")
        components = _require_tuple(self.artifact_components, "artifact_components")
        for component in components:
            if type(component) is not FrvA2ArtifactComponentBinding:
                raise ValueError(
                    "artifact_components must contain FrvA2ArtifactComponentBinding"
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
        _require_sha(
            self.geometry_point_content_digest, "geometry_point_content_digest"
        )
        for name in (
            "valid_count",
            "insufficient_count",
            "degenerate_count",
            "neighborhood_sufficient_count",
            "density_supported_count",
            "canopy_branching_compatible_count",
            "woody_column_compatible_count",
            "ground_surface_compatible_count",
            "vertical_continuity_count",
            "neighbor_cap_reached_count",
            "support_neighbor_cap_reached_count",
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
            raise ValueError(
                "proof_digest does not match the canonical P-GEOMETRY proof fields"
            )

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": FRV_GEOMETRY_PROOF_DIGEST_DOMAIN,
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
            "geometry_point_content_digest": self.geometry_point_content_digest,
            "valid_count": self.valid_count,
            "insufficient_count": self.insufficient_count,
            "degenerate_count": self.degenerate_count,
            "neighborhood_sufficient_count": self.neighborhood_sufficient_count,
            "density_supported_count": self.density_supported_count,
            "canopy_branching_compatible_count": self.canopy_branching_compatible_count,
            "woody_column_compatible_count": self.woody_column_compatible_count,
            "ground_surface_compatible_count": self.ground_surface_compatible_count,
            "vertical_continuity_count": self.vertical_continuity_count,
            "neighbor_cap_reached_count": self.neighbor_cap_reached_count,
            "support_neighbor_cap_reached_count": self.support_neighbor_cap_reached_count,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_xyz": list(self.origin_xyz),
            "transform_declaration": self.transform_declaration,
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
    def from_dict(cls, data: Mapping[str, Any]) -> FrvGeometryProviderProof:
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
        values["error_codes"] = tuple(
            _enum_from_value(item, FrvErrorCode, "error_codes[]")
            for item in _tuple_from_json(values["error_codes"], "error_codes")
        )
        return cls(**values)


def _validate_proof_structure(proof: FrvGeometryProviderProof) -> None:
    if type(proof) is not FrvGeometryProviderProof:
        raise ValueError("proof must be a FrvGeometryProviderProof")
    if proof.schema_version != FRV_GEOMETRY_PROOF_SCHEMA_VERSION:
        raise ValueError("schema_version must match the frozen proof schema version")
    if proof.provider_key != FRV_GEOMETRY_PROVIDER_KEY:
        raise ValueError("provider_key must be geometry_provider")
    if type(proof.provider_role) is not FrvA2ProviderRole:
        raise ValueError("provider_role must be a FrvA2ProviderRole")
    if proof.provider_role is not FrvA2ProviderRole.GEOMETRY:
        raise ValueError("provider_role must be GEOMETRY")
    if proof.interface_version != FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION:
        raise ValueError("interface_version must match the frozen interface version")
    if proof.algorithm_version != FRV_GEOMETRY_ALGORITHM_VERSION:
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
    _require_sha(
        proof.geometry_point_content_digest, "geometry_point_content_digest"
    )
    for name in (
        "valid_count",
        "insufficient_count",
        "degenerate_count",
        "neighborhood_sufficient_count",
        "density_supported_count",
        "canopy_branching_compatible_count",
        "woody_column_compatible_count",
        "ground_surface_compatible_count",
        "vertical_continuity_count",
        "neighbor_cap_reached_count",
        "support_neighbor_cap_reached_count",
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
    if proof.quality_flags != 0 or proof.reason_flags != 0:
        raise ValueError("success proof must have NONE quality and reason flags")
    error_codes = _require_tuple(proof.error_codes, "error_codes")
    for code in error_codes:
        _require_enum(code, FrvErrorCode, "error_codes[]")
    if error_codes != ():
        raise ValueError("success proof must have empty error_codes")
    _require_sha(proof.proof_digest, "proof_digest")
    if proof.proof_digest != proof.computed_digest():
        raise ValueError("proof_digest does not match the canonical P-GEOMETRY proof fields")


def validate_frv_geometry_provider_proof(proof: FrvGeometryProviderProof) -> None:
    _validate_proof_structure(proof)
    state_sum = (
        proof.valid_count + proof.insufficient_count + proof.degenerate_count
    )
    if state_sum != proof.source_point_count:
        raise ValueError("evidence state counts must sum to source_point_count")
    if proof.neighborhood_sufficient_count != proof.valid_count:
        raise ValueError("neighborhood_sufficient_count must equal valid_count")
    for name in (
        "density_supported_count",
        "canopy_branching_compatible_count",
        "woody_column_compatible_count",
        "ground_surface_compatible_count",
        "vertical_continuity_count",
    ):
        if getattr(proof, name) > proof.source_point_count:
            raise ValueError(f"{name} must not exceed source_point_count")
    if (
        proof.canopy_branching_compatible_count > proof.valid_count
        or proof.woody_column_compatible_count > proof.valid_count
        or proof.ground_surface_compatible_count > proof.valid_count
    ):
        raise ValueError("VALID-gated evidence counts must not exceed valid_count")
    if proof.neighbor_cap_reached_count > proof.source_point_count:
        raise ValueError("neighbor_cap_reached_count must not exceed source_point_count")
    if proof.support_neighbor_cap_reached_count > proof.source_point_count:
        raise ValueError(
            "support_neighbor_cap_reached_count must not exceed source_point_count"
        )
    expected_index_digest = _stream_index_domain_digest(
        proof.domain_count, proof.source_content_identity_digest
    )
    if proof.original_index_domain_digest != expected_index_digest:
        raise ValueError("original_index_domain_digest does not match the FULL_SOURCE domain")
    if proof.proof_digest != proof.computed_digest():
        raise ValueError("proof_digest does not match the canonical P-GEOMETRY proof fields")


# ---------------------------------------------------------------------------
# Finalized P-GEOMETRY result with private capability
# ---------------------------------------------------------------------------

_MISSING = object()


class _FRV_GEOMETRY_FINALIZED_CAPABILITY:
    """Module-private identity-only finalized P-GEOMETRY capability token."""

    __slots__ = ()


@dataclass(frozen=True, slots=True, init=False)
class FrvGeometryProviderResult:
    """Non-serializable finalized P-GEOMETRY success wrapper (private capability)."""

    proof: FrvGeometryProviderProof
    source_identity: SourceIdentity
    _finalized_capability: Any = field(init=False, repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrvGeometryProviderResult has no public constructor")


def _validate_finalized_geometry_bindings(
    proof: FrvGeometryProviderProof, source_identity: SourceIdentity
) -> None:
    validate_frv_geometry_provider_proof(proof)
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


def _create_finalized_frv_geometry_provider_result(
    proof: FrvGeometryProviderProof,
    source_identity: SourceIdentity,
    *,
    capability: Any,
) -> FrvGeometryProviderResult:
    if type(proof) is not FrvGeometryProviderProof:
        raise TypeError("proof must be a FrvGeometryProviderProof")
    if type(source_identity) is not SourceIdentity:
        raise TypeError("source_identity must be a SourceIdentity")
    if capability is not _FRV_GEOMETRY_FINALIZED_CAPABILITY:
        raise ValueError("capability must be the module-private finalized capability")
    _validate_finalized_geometry_bindings(proof, source_identity)
    result = object.__new__(FrvGeometryProviderResult)
    object.__setattr__(result, "proof", proof)
    object.__setattr__(result, "source_identity", source_identity)
    object.__setattr__(result, "_finalized_capability", capability)
    return result


def validate_finalized_frv_geometry_provider_result(
    result: FrvGeometryProviderResult,
) -> None:
    if type(result) is not FrvGeometryProviderResult:
        raise TypeError("result must be a FrvGeometryProviderResult")
    if getattr(result, "_finalized_capability", _MISSING) is not (
        _FRV_GEOMETRY_FINALIZED_CAPABILITY
    ):
        raise ValueError("result must carry the module-private finalized capability")
    proof = getattr(result, "proof", _MISSING)
    if type(proof) is not FrvGeometryProviderProof:
        raise ValueError("result.proof must be a FrvGeometryProviderProof")
    source_identity = getattr(result, "source_identity", _MISSING)
    if type(source_identity) is not SourceIdentity:
        raise ValueError("result.source_identity must be a SourceIdentity")
    _validate_finalized_geometry_bindings(proof, source_identity)


# ---------------------------------------------------------------------------
# FrvGeometryLoadError
# ---------------------------------------------------------------------------


class FrvGeometryLoadError(Exception):
    """P-GEOMETRY load failure carrying a frozen A2 failure record."""

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
# FrvGeometryLoadSession protocol
# ---------------------------------------------------------------------------


class FrvGeometryLoadSession(Protocol):
    """Single-use P-GEOMETRY load session interface (contracts only)."""

    def iter_chunks(self) -> Iterator[FrvGeometryChunk]: ...
    def finalize(self) -> FrvGeometryProviderResult: ...
    def close(self) -> None: ...
    def __enter__(self) -> FrvGeometryLoadSession: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]: ...


# ---------------------------------------------------------------------------
# FRVA2BND1 adapter
# ---------------------------------------------------------------------------


def to_frv_a2_geometry_provider_binding_evidence(
    result: FrvGeometryProviderResult,
) -> FrvA2ProviderBindingEvidence:
    """Return the success-only FRVA2BND1 formal provider evidence for P-GEOMETRY."""

    validate_finalized_frv_geometry_provider_result(result)
    proof = result.proof
    evidence = FrvA2ProviderBindingEvidence.__new__(FrvA2ProviderBindingEvidence)
    object.__setattr__(evidence, "provider_key", FRV_GEOMETRY_PROVIDER_KEY)
    object.__setattr__(evidence, "provider_role", FrvA2ProviderRole.GEOMETRY)
    object.__setattr__(evidence, "evidence_class", FrvA2EvidenceClass.FORMAL_PROVIDER)
    object.__setattr__(evidence, "public_module", "algorithms.vegetation.geometry_provider")
    object.__setattr__(evidence, "public_type_name", "FrvGeometryProviderResult")
    object.__setattr__(evidence, "public_loader_name", "load_frv_geometry")
    object.__setattr__(
        evidence, "interface_version", FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION
    )
    object.__setattr__(evidence, "schema_version", FRV_GEOMETRY_PROVIDER_SCHEMA_VERSION)
    object.__setattr__(evidence, "algorithm_version", FRV_GEOMETRY_ALGORITHM_VERSION)
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
                proof.geometry_point_content_digest,
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
    "FRV_GEOMETRY_PROVIDER_KEY",
    "FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION",
    "FRV_GEOMETRY_PROVIDER_SCHEMA_VERSION",
    "FRV_GEOMETRY_CONFIG_SCHEMA_VERSION",
    "FRV_GEOMETRY_PROOF_SCHEMA_VERSION",
    "FRV_GEOMETRY_ALGORITHM_VERSION",
    "FRV_GEOMETRY_CONFIG_DIGEST_DOMAIN",
    "FRV_GEOMETRY_POINT_CONTENT_DIGEST_DOMAIN",
    "FRV_GEOMETRY_PROOF_DIGEST_DOMAIN",
    "FRV_GEOMETRY_CACHE_BYTES",
    "FrvGeometryEvidenceFlag",
    "FrvGeometryEvidenceState",
    "FrvGeometryReasonFlag",
    "FrvGeometryQualityFlag",
    "FrvGeometryProviderConfig",
    "FrvGeometryChunk",
    "FrvGeometryProviderProof",
    "FrvGeometryProviderResult",
    "FrvGeometryLoadError",
    "FrvGeometryLoadSession",
    "validate_frv_geometry_provider_config",
    "validate_frv_geometry_chunk",
    "validate_frv_geometry_provider_proof",
    "validate_finalized_frv_geometry_provider_result",
    "derive_frv_geometry_provider_config_digest",
    "frv_geometry_point_content_digest",
    "to_frv_a2_geometry_provider_binding_evidence",
]
