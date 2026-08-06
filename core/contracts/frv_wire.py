"""FRV-A2 P-WIRE provider contracts (PLAN-FROZEN).

P-WIRE consumes a finalized P-SOURCE result and an approved, source-bound
``FrvWireAuthorityManifest`` and emits full-resolution point-to-polyline
proximity evidence over the complete FULL_SOURCE domain.  This module freezes
the authority manifest/reference, provider config, runtime chunk, success
proof, finalized result, load error, session protocol, validators, digest
framings (``FRVWIRAUTH1``, ``FRVWIRGEO1``, ``FRVWIRCFG1``, ``FRVWIRPCD1``,
``FRVWIRPRF1``) and the ``FRVA2BND1`` adapter.

The point relation enum deliberately has no ``WIRE_BODY_CONFIRMED`` value:
confirmation is a later V4 concern, never a P-WIRE result.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol

import numpy as np

from core.contracts.frv import (
    FrvErrorCode,
    FrvQualityFlag,
    FrvWireProviderState,
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
    _legacy_pair_layout,
    frv_a2_provider_binding_digest,
    validate_frv_a2_failure_record,
)

# ---------------------------------------------------------------------------
# Public constants (frozen names and values)
# ---------------------------------------------------------------------------

FRV_WIRE_PROVIDER_KEY = "wire_provider"
FRV_WIRE_PROVIDER_INTERFACE_VERSION = "frv-wire-provider-v1"
FRV_WIRE_PROVIDER_SCHEMA_VERSION = "frv-wire-provider-result-v1"
FRV_WIRE_CONFIG_SCHEMA_VERSION = "frv-wire-provider-config-v1"
FRV_WIRE_PROOF_SCHEMA_VERSION = "frv-wire-provider-proof-v1"
FRV_WIRE_ALGORITHM_VERSION = "frv-wire-polyline-distance-grid-v1"
FRV_WIRE_CONFIG_DIGEST_DOMAIN = "FRVWIRCFG1"
FRV_WIRE_POINT_CONTENT_DIGEST_DOMAIN = "FRVWIRPCD1"
FRV_WIRE_PROOF_DIGEST_DOMAIN = "FRVWIRPRF1"

FRV_WIRE_AUTHORITY_SCHEMA_VERSION = "frv-wire-authority-v1"
FRV_WIRE_AUTHORITY_DIGEST_DOMAIN = "FRVWIRAUTH1"
FRV_WIRE_GEOMETRY_DIGEST_DOMAIN = "FRVWIRGEO1"

# Frozen default operational parameters (plan section 4.2).
FRV_WIRE_DEFAULT_BODY_POSSIBLE_RADIUS_M = 0.10
FRV_WIRE_DEFAULT_CONFLICT_RADIUS_M = 0.50
FRV_WIRE_DEFAULT_PROXIMITY_RADIUS_M = 1.50
FRV_WIRE_DEFAULT_SPATIAL_INDEX_CELL_SIZE_M = 1.50
FRV_WIRE_DEFAULT_MAX_EDGES_PER_INDEX_CELL = 100000

# Digest domain that binds the exact serialization of the P-SOURCE proof into
# the P-WIRE proof (same framing shape as the P-HAG source-proof binding).
_FRV_WIRE_SOURCE_PROOF_DIGEST_DOMAIN = "FRVWIRSRC1"

_FRV_WIRE_ARTIFACT_REFERENCE_PREFIX = "frv-a2-ref-v1|"

# ---------------------------------------------------------------------------
# Frozen public enums
# ---------------------------------------------------------------------------


class FrvWireAuthorityMode(str, Enum):
    """The only two provable wire-authority states."""

    W1_MODEL_SET = "w1_model_set"
    EMPTY_INVENTORY = "empty_inventory"


class FrvWirePointRelationState(IntEnum):
    """Closed per-point wire relation; no body-confirmed value exists."""

    OUTSIDE_INFLUENCE_DOMAIN = 0
    WIRE_BODY_POSSIBLE = 2
    WIRE_PROXIMITY_ONLY = 3
    WIRE_VEGETATION_CONFLICT = 4


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


def _require_artifact_reference_id(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if not text.startswith(_FRV_WIRE_ARTIFACT_REFERENCE_PREFIX):
        raise ValueError(f"{name} must start with {_FRV_WIRE_ARTIFACT_REFERENCE_PREFIX!r}")
    return text


def _normalize_zero(value: float) -> float:
    """Return +0.0 for any exact-zero float, otherwise the finite value."""
    return 0.0 if value == 0.0 else float(value)


def _is_negative_zero(value: float) -> bool:
    return value == 0.0 and math.copysign(1.0, value) < 0.0


def _strictly_increasing_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    items = _require_tuple(value, name)
    validated: list[int] = []
    previous: int | None = None
    for item in items:
        integer = _require_nonnegative_int(item, f"{name}[]")
        if previous is not None and integer <= previous:
            raise ValueError(f"{name} must be strictly increasing")
        validated.append(integer)
        previous = integer
    return tuple(validated)


def _component_sort_key(component: FrvA2ArtifactComponentBinding) -> tuple[int, str]:
    order = {
        member: index
        for index, member in enumerate(FrvA2ArtifactComponentKind)
    }
    return (order[component.component_kind], component.component_key)


def _validate_w1_components(
    components: tuple[FrvA2ArtifactComponentBinding, ...], name: str
) -> None:
    _legacy_pair_layout(components, name)


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


# ---------------------------------------------------------------------------
# Authority manifest digest helpers
# ---------------------------------------------------------------------------


def derive_frv_wire_authority_approval_digest(
    manifest: FrvWireAuthorityManifest,
) -> str:
    """Return the approval digest over every canonical field except the two self-digests."""

    payload = manifest.to_dict()
    payload.pop("approval_digest", None)
    payload.pop("manifest_digest", None)
    return hashlib.sha256(canonical_frv_json_bytes(payload)).hexdigest()


def derive_frv_wire_authority_manifest_digest(
    manifest: FrvWireAuthorityManifest,
) -> str:
    """Return the FRVWIRAUTH1 manifest digest over every field except itself."""

    payload: dict[str, Any] = {
        "digest_domain": FRV_WIRE_AUTHORITY_DIGEST_DOMAIN,
        "schema_version": manifest.schema_version,
        "authority_mode": manifest.authority_mode.value,
        "source_content_identity_digest": manifest.source_content_identity_digest,
        "source_identity_record_digest": manifest.source_identity_record_digest,
        "source_id": manifest.source_id,
        "coordinate_frame": manifest.coordinate_frame,
        "units": manifest.units,
        "origin_xyz": list(manifest.origin_xyz),
        "transform_declaration": manifest.transform_declaration,
        "inventory_complete": manifest.inventory_complete,
        "approved_wire_ids": list(manifest.approved_wire_ids),
        "w1_artifact_components": [
            component.to_dict() for component in manifest.w1_artifact_components
        ],
        "geometry_content_digest": manifest.geometry_content_digest,
        "authority_rule_id": manifest.authority_rule_id,
        "authority_rule_version": manifest.authority_rule_version,
        "approver": manifest.approver,
        "approval_timestamp": manifest.approval_timestamp,
        "approval_digest": manifest.approval_digest,
    }
    return frv_digest(payload)


# ---------------------------------------------------------------------------
# FrvWireAuthorityManifest
# ---------------------------------------------------------------------------


def _check_wire_authority_manifest(manifest: FrvWireAuthorityManifest) -> None:
    _require_exact_str(manifest.schema_version, "schema_version")
    if manifest.schema_version != FRV_WIRE_AUTHORITY_SCHEMA_VERSION:
        raise ValueError("unsupported wire authority schema version")
    _require_enum(manifest.authority_mode, FrvWireAuthorityMode, "authority_mode")
    _require_sha(
        manifest.source_content_identity_digest, "source_content_identity_digest"
    )
    _require_sha(
        manifest.source_identity_record_digest, "source_identity_record_digest"
    )
    _require_exact_str(manifest.source_id, "source_id")
    _require_exact_str(manifest.coordinate_frame, "coordinate_frame")
    if manifest.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
        raise ValueError("coordinate_frame must be LOCAL_CARTESIAN_XYZ")
    _require_exact_str(manifest.units, "units")
    if manifest.units != "m":
        raise ValueError("units must be m")
    object.__setattr__(
        manifest, "origin_xyz", _validate_vec3(manifest.origin_xyz, "origin_xyz")
    )
    _require_exact_str(manifest.transform_declaration, "transform_declaration")
    if manifest.transform_declaration != "identity":
        raise ValueError("transform_declaration must be identity")
    if type(manifest.inventory_complete) is not bool:
        raise ValueError("inventory_complete must be a built-in bool")
    if manifest.inventory_complete is not True:
        raise ValueError("an approved wire authority must have inventory_complete=True")
    object.__setattr__(
        manifest,
        "approved_wire_ids",
        _strictly_increasing_int_tuple(manifest.approved_wire_ids, "approved_wire_ids"),
    )
    components = _require_tuple(manifest.w1_artifact_components, "w1_artifact_components")
    for component in components:
        if type(component) is not FrvA2ArtifactComponentBinding:
            raise ValueError(
                "w1_artifact_components must contain FrvA2ArtifactComponentBinding"
            )
    if manifest.authority_mode is FrvWireAuthorityMode.W1_MODEL_SET:
        if len(components) != 2:
            raise ValueError("W1_MODEL_SET requires the exact W1 JSON/NPZ pair")
        _validate_w1_components(components, "w1_artifact_components")
        if not manifest.approved_wire_ids:
            raise ValueError("W1_MODEL_SET requires a non-empty approved wire inventory")
    else:  # EMPTY_INVENTORY
        if components != ():
            raise ValueError("EMPTY_INVENTORY requires no W1 artifact components")
        if manifest.approved_wire_ids != ():
            raise ValueError("EMPTY_INVENTORY requires an empty approved wire inventory")
    _require_sha(manifest.geometry_content_digest, "geometry_content_digest")
    for name in (
        "authority_rule_id",
        "authority_rule_version",
        "approver",
        "approval_timestamp",
    ):
        _require_exact_str(getattr(manifest, name), name)
    _require_sha(manifest.approval_digest, "approval_digest")
    _require_sha(manifest.manifest_digest, "manifest_digest")
    if manifest.approval_digest != derive_frv_wire_authority_approval_digest(manifest):
        raise ValueError("approval_digest does not match the canonical manifest fields")
    if manifest.manifest_digest != derive_frv_wire_authority_manifest_digest(manifest):
        raise ValueError("manifest_digest does not match the canonical FRVWIRAUTH1 fields")


@dataclass(frozen=True, slots=True)
class FrvWireAuthorityManifest:
    """Immutable approved, source-bound wire authority manifest."""

    schema_version: str
    authority_mode: FrvWireAuthorityMode
    source_content_identity_digest: str
    source_identity_record_digest: str
    source_id: str
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    inventory_complete: bool
    approved_wire_ids: tuple[int, ...]
    w1_artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    geometry_content_digest: str
    authority_rule_id: str
    authority_rule_version: str
    approver: str
    approval_timestamp: str
    approval_digest: str
    manifest_digest: str

    def __post_init__(self) -> None:
        _check_wire_authority_manifest(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "authority_mode": self.authority_mode.value,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "source_id": self.source_id,
            "coordinate_frame": self.coordinate_frame,
            "units": self.units,
            "origin_xyz": list(self.origin_xyz),
            "transform_declaration": self.transform_declaration,
            "inventory_complete": self.inventory_complete,
            "approved_wire_ids": list(self.approved_wire_ids),
            "w1_artifact_components": [
                component.to_dict() for component in self.w1_artifact_components
            ],
            "geometry_content_digest": self.geometry_content_digest,
            "authority_rule_id": self.authority_rule_id,
            "authority_rule_version": self.authority_rule_version,
            "approver": self.approver,
            "approval_timestamp": self.approval_timestamp,
            "approval_digest": self.approval_digest,
            "manifest_digest": self.manifest_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvWireAuthorityManifest:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["authority_mode"] = _enum_from_value(
            values["authority_mode"], FrvWireAuthorityMode, "authority_mode"
        )
        values["approved_wire_ids"] = _tuple_from_json(
            values["approved_wire_ids"], "approved_wire_ids"
        )
        values["w1_artifact_components"] = tuple(
            FrvA2ArtifactComponentBinding.from_dict(item)
            for item in _tuple_from_json(
                values["w1_artifact_components"], "w1_artifact_components"
            )
        )
        values["origin_xyz"] = _tuple_from_json(values["origin_xyz"], "origin_xyz")
        return cls(**values)


def validate_frv_wire_authority_manifest(
    manifest: FrvWireAuthorityManifest,
) -> None:
    if type(manifest) is not FrvWireAuthorityManifest:
        raise ValueError("manifest must be a FrvWireAuthorityManifest")
    _check_wire_authority_manifest(manifest)
    if manifest.manifest_digest != derive_frv_wire_authority_manifest_digest(manifest):
        raise ValueError("manifest_digest does not match the canonical FRVWIRAUTH1 fields")


# ---------------------------------------------------------------------------
# FrvWireAuthorityReference (non-serializable runtime object)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvWireAuthorityReference:
    """Runtime-only binding between an authority manifest and absolute paths.

    Paths never enter any digest.  The manifest component is the
    ``MANIFEST_JSON`` / ``authority_manifest_json`` component that binds the
    authority JSON into the formal P-WIRE artifact tuple.
    """

    manifest: FrvWireAuthorityManifest
    authority_manifest_path: Path
    authority_manifest_component: FrvA2ArtifactComponentBinding
    w1_json_path: Path | None
    w1_npz_path: Path | None

    def __post_init__(self) -> None:
        if type(self.manifest) is not FrvWireAuthorityManifest:
            raise ValueError("manifest must be a FrvWireAuthorityManifest")
        if not isinstance(self.authority_manifest_path, Path):
            raise ValueError("authority_manifest_path must be a pathlib.Path instance")
        if type(self.authority_manifest_component) is not FrvA2ArtifactComponentBinding:
            raise ValueError(
                "authority_manifest_component must be a FrvA2ArtifactComponentBinding"
            )
        component = self.authority_manifest_component
        if component.component_kind is not FrvA2ArtifactComponentKind.MANIFEST_JSON:
            raise ValueError("authority_manifest_component must be kind MANIFEST_JSON")
        if component.component_key != "authority_manifest_json":
            raise ValueError("authority_manifest_component key must be authority_manifest_json")
        if self.manifest.authority_mode is FrvWireAuthorityMode.W1_MODEL_SET:
            if not isinstance(self.w1_json_path, Path) or not isinstance(
                self.w1_npz_path, Path
            ):
                raise ValueError("W1_MODEL_SET requires the W1 JSON and NPZ paths")
        else:
            if self.w1_json_path is not None or self.w1_npz_path is not None:
                raise ValueError("EMPTY_INVENTORY forbids W1 paths")


# ---------------------------------------------------------------------------
# FRVWIRCFG1 provider-config digest and config
# ---------------------------------------------------------------------------


def derive_frv_wire_provider_config_digest(
    *,
    schema_version: str,
    provider_key: str,
    provider_role: FrvA2ProviderRole,
    interface_version: str,
    algorithm_version: str,
    wire_body_possible_radius_m: float,
    vegetation_conflict_radius_m: float,
    proximity_radius_m: float,
    spatial_index_cell_size_m: float,
    max_edges_per_index_cell: int,
) -> str:
    """Return the FRVWIRCFG1 digest for the frozen provider-config payload."""

    payload = {
        "digest_domain": FRV_WIRE_CONFIG_DIGEST_DOMAIN,
        "schema_version": schema_version,
        "provider_key": provider_key,
        "provider_role": provider_role.value,
        "interface_version": interface_version,
        "algorithm_version": algorithm_version,
        "wire_body_possible_radius_m": wire_body_possible_radius_m,
        "vegetation_conflict_radius_m": vegetation_conflict_radius_m,
        "proximity_radius_m": proximity_radius_m,
        "spatial_index_cell_size_m": spatial_index_cell_size_m,
        "max_edges_per_index_cell": max_edges_per_index_cell,
    }
    return frv_digest(payload)


@dataclass(frozen=True, slots=True)
class FrvWireProviderConfig:
    """Exact-schema P-WIRE provider configuration (digest domain FRVWIRCFG1)."""

    schema_version: str
    provider_key: str
    provider_role: FrvA2ProviderRole
    interface_version: str
    algorithm_version: str
    wire_body_possible_radius_m: float
    vegetation_conflict_radius_m: float
    proximity_radius_m: float
    spatial_index_cell_size_m: float
    max_edges_per_index_cell: int
    provider_config_digest: str

    def __post_init__(self) -> None:
        _require_exact_str(self.schema_version, "schema_version")
        if self.schema_version != FRV_WIRE_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported P-WIRE provider config schema version")
        _require_exact_str(self.provider_key, "provider_key")
        if self.provider_key != FRV_WIRE_PROVIDER_KEY:
            raise ValueError("provider_key must be wire_provider")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        if self.provider_role is not FrvA2ProviderRole.WIRE:
            raise ValueError("provider_role must be WIRE")
        _require_exact_str(self.interface_version, "interface_version")
        if self.interface_version != FRV_WIRE_PROVIDER_INTERFACE_VERSION:
            raise ValueError("interface_version must match the frozen interface version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != FRV_WIRE_ALGORITHM_VERSION:
            raise ValueError("algorithm_version must match the frozen algorithm version")
        body = _require_finite_float(
            self.wire_body_possible_radius_m, "wire_body_possible_radius_m"
        )
        object.__setattr__(self, "wire_body_possible_radius_m", body)
        conflict = _require_finite_float(
            self.vegetation_conflict_radius_m, "vegetation_conflict_radius_m"
        )
        object.__setattr__(self, "vegetation_conflict_radius_m", conflict)
        proximity = _require_finite_float(
            self.proximity_radius_m, "proximity_radius_m"
        )
        object.__setattr__(self, "proximity_radius_m", proximity)
        if body <= 0.0 or not (body < conflict < proximity):
            raise ValueError("wire radii must be strictly increasing positive values")
        object.__setattr__(
            self,
            "spatial_index_cell_size_m",
            _require_finite_float(self.spatial_index_cell_size_m, "spatial_index_cell_size_m"),
        )
        if self.spatial_index_cell_size_m <= 0.0:
            raise ValueError("spatial_index_cell_size_m must be positive")
        object.__setattr__(
            self,
            "max_edges_per_index_cell",
            _require_positive_int(self.max_edges_per_index_cell, "max_edges_per_index_cell"),
        )
        _require_sha(self.provider_config_digest, "provider_config_digest")
        computed = derive_frv_wire_provider_config_digest(
            schema_version=self.schema_version,
            provider_key=self.provider_key,
            provider_role=self.provider_role,
            interface_version=self.interface_version,
            algorithm_version=self.algorithm_version,
            wire_body_possible_radius_m=self.wire_body_possible_radius_m,
            vegetation_conflict_radius_m=self.vegetation_conflict_radius_m,
            proximity_radius_m=self.proximity_radius_m,
            spatial_index_cell_size_m=self.spatial_index_cell_size_m,
            max_edges_per_index_cell=self.max_edges_per_index_cell,
        )
        if self.provider_config_digest != computed:
            raise ValueError(
                "provider_config_digest does not match canonical FRVWIRCFG1 fields"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider_key": self.provider_key,
            "provider_role": self.provider_role.value,
            "interface_version": self.interface_version,
            "algorithm_version": self.algorithm_version,
            "wire_body_possible_radius_m": self.wire_body_possible_radius_m,
            "vegetation_conflict_radius_m": self.vegetation_conflict_radius_m,
            "proximity_radius_m": self.proximity_radius_m,
            "spatial_index_cell_size_m": self.spatial_index_cell_size_m,
            "max_edges_per_index_cell": self.max_edges_per_index_cell,
            "provider_config_digest": self.provider_config_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FrvWireProviderConfig:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_role"] = _enum_from_value(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        return cls(**values)


def validate_frv_wire_provider_config(config: FrvWireProviderConfig) -> None:
    if type(config) is not FrvWireProviderConfig:
        raise ValueError("config must be a FrvWireProviderConfig")
    computed = derive_frv_wire_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        wire_body_possible_radius_m=config.wire_body_possible_radius_m,
        vegetation_conflict_radius_m=config.vegetation_conflict_radius_m,
        proximity_radius_m=config.proximity_radius_m,
        spatial_index_cell_size_m=config.spatial_index_cell_size_m,
        max_edges_per_index_cell=config.max_edges_per_index_cell,
    )
    if config.provider_config_digest != computed:
        raise ValueError("provider_config_digest does not match canonical FRVWIRCFG1 fields")


# ---------------------------------------------------------------------------
# FrvWireChunk (runtime-only immutable contract)
# ---------------------------------------------------------------------------

_FRV_WIRE_CHUNK_ARRAY_SPECS = (
    ("original_index", "<i8", (), False, False),
    ("provider_state", "|u1", (), False, False),
    ("relation_state", "|u1", (), False, False),
    ("nearest_wire_id", "<i4", (), False, False),
    ("nearest_segment_index", "<i4", (), False, False),
    ("nearest_edge_index", "<i4", (), False, False),
    ("centerline_distance_m", "<f8", (), True, True),
    ("reason_flags", "<u4", (), False, False),
    ("quality_flags", "<u4", (), False, False),
)


@dataclass(frozen=True, slots=True)
class FrvWireChunk:
    """Read-only P-WIRE chunk with a one-to-one source-chunk mapping."""

    source_id: str
    chunk_id: int
    start_index: int
    point_count: int
    original_index: np.ndarray
    provider_state: np.ndarray
    relation_state: np.ndarray
    nearest_wire_id: np.ndarray
    nearest_segment_index: np.ndarray
    nearest_edge_index: np.ndarray
    centerline_distance_m: np.ndarray
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
        for name, dtype, shape, finite, zero_normalized in _FRV_WIRE_CHUNK_ARRAY_SPECS:
            _require_ndarray(
                getattr(self, name),
                name,
                dtype=np.dtype(dtype),
                shape=(point_count,) + shape,
                finite=finite,
                zero_normalized=zero_normalized,
            )


_VALID_PROVIDER_STATES = frozenset(
    {
        int(FrvWireProviderState.WIRE_RELATION_VALID),
        int(FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN),
    }
)
_VALID_RELATION_STATES = frozenset(
    {
        int(FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN),
        int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE),
        int(FrvWirePointRelationState.WIRE_PROXIMITY_ONLY),
        int(FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT),
    }
)


def validate_frv_wire_chunk(
    chunk: FrvWireChunk,
    *,
    provider_config: FrvWireProviderConfig,
) -> None:
    """Validate the frozen per-row wire semantics against the provider config."""

    if type(chunk) is not FrvWireChunk:
        raise ValueError("chunk must be a FrvWireChunk")
    if type(provider_config) is not FrvWireProviderConfig:
        raise ValueError("provider_config must be a FrvWireProviderConfig")
    r0 = float(provider_config.wire_body_possible_radius_m)
    r1 = float(provider_config.vegetation_conflict_radius_m)
    r2 = float(provider_config.proximity_radius_m)
    previous: int | None = None
    for row in range(chunk.point_count):
        index = int(chunk.original_index[row])
        if index < 0:
            raise ValueError("original_index must be nonnegative")
        if previous is not None and index <= previous:
            raise ValueError("original_index must be strictly increasing within chunk")
        previous = index
        provider_state = int(chunk.provider_state[row])
        if provider_state not in _VALID_PROVIDER_STATES:
            raise ValueError("provider_state must be a FrvWireProviderState value")
        relation = int(chunk.relation_state[row])
        if relation not in _VALID_RELATION_STATES:
            raise ValueError("relation_state must be a FrvWirePointRelationState value")
        wire_id = int(chunk.nearest_wire_id[row])
        segment_index = int(chunk.nearest_segment_index[row])
        edge_index = int(chunk.nearest_edge_index[row])
        distance = float(chunk.centerline_distance_m[row])
        in_domain = relation in {
            int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE),
            int(FrvWirePointRelationState.WIRE_PROXIMITY_ONLY),
            int(FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT),
        }
        if provider_state == int(FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN):
            if relation != int(FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN):
                raise ValueError("not-applicable rows must carry OUTSIDE relation")
            if wire_id != -1 or segment_index != -1 or edge_index != -1:
                raise ValueError("not-applicable rows must carry -1 wire/segment/edge ids")
            if distance != 0.0:
                raise ValueError("not-applicable rows must carry positive-zero distance")
            continue
        if not in_domain:
            if wire_id != -1 or segment_index != -1 or edge_index != -1:
                raise ValueError("OUTSIDE rows must carry -1 wire/segment/edge ids")
            if distance != 0.0:
                raise ValueError("OUTSIDE rows must carry positive-zero sentinel distance")
            continue
        if wire_id < 0 or segment_index < 0 or edge_index < 0:
            raise ValueError("in-domain rows require nonnegative wire/segment/edge ids")
        if distance < 0.0:
            raise ValueError("in-domain centerline distance must be nonnegative")
        if relation == int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE):
            if not (0.0 <= distance <= r0):
                raise ValueError("WIRE_BODY_POSSIBLE distance must lie in [0, body radius]")
        elif relation == int(FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT):
            if not (r0 < distance <= r1):
                raise ValueError(
                    "WIRE_VEGETATION_CONFLICT distance must lie in (body, conflict]"
                )
        else:  # WIRE_PROXIMITY_ONLY
            if not (r1 < distance <= r2):
                raise ValueError("WIRE_PROXIMITY_ONLY distance must lie in (conflict, proximity]")


# ---------------------------------------------------------------------------
# FRVWIRGEO1 wire geometry content digest
# ---------------------------------------------------------------------------


def frv_wire_geometry_content_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    origin_xyz: tuple[float, float, float],
    edges: Iterable[tuple[int, int, int, float, float, float, float, float, float]],
) -> str:
    """Return the FRVWIRGEO1 digest over sorted unique polyline edges.

    Edges are the closed segments of the sampled polylines in absolute
    coordinates (``float64(sampled_curve_xyz) + origin_xyz``), ordered by
    ``(wire_id, segment_index, edge_index)``.
    """

    _require_sha(source_content_identity_digest, "source_content_identity_digest")
    _require_sha(source_identity_record_digest, "source_identity_record_digest")
    origin = _validate_vec3(origin_xyz, "origin_xyz")
    sorted_edges: list[tuple[int, int, int, float, float, float, float, float, float]] = []
    previous: tuple[int, int, int] | None = None
    for edge in edges:
        if len(edge) != 9:
            raise ValueError("edges must be (wire_id, segment_index, edge_index, a, b) tuples")
        wire_id, segment_index, edge_index = edge[0], edge[1], edge[2]
        if (
            type(wire_id) is not int
            or type(segment_index) is not int
            or type(edge_index) is not int
        ):
            raise ValueError("edge ids must be built-in ints")
        if wire_id < 0 or segment_index < 0 or edge_index < 0:
            raise ValueError("edge ids must be nonnegative")
        ax = _require_finite_float(edge[3], "edges[][a0]")
        ay = _require_finite_float(edge[4], "edges[][a1]")
        az = _require_finite_float(edge[5], "edges[][a2]")
        bx = _require_finite_float(edge[6], "edges[][b0]")
        by = _require_finite_float(edge[7], "edges[][b1]")
        bz = _require_finite_float(edge[8], "edges[][b2]")
        key = (wire_id, segment_index, edge_index)
        if previous is not None and key <= previous:
            raise ValueError("edges must be strictly ordered by (wire_id, segment, edge)")
        previous = key
        sorted_edges.append((wire_id, segment_index, edge_index, ax, ay, az, bx, by, bz))
    hasher = hashlib.sha256()
    hasher.update(b"FRVWIRGEO1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<ddd", origin[0], origin[1], origin[2]))
    hasher.update(struct.pack("<Q", len(sorted_edges)))
    for wire_id, segment_index, edge_index, ax, ay, az, bx, by, bz in sorted_edges:
        hasher.update(
            struct.pack(
                "<iii6d",
                wire_id,
                segment_index,
                edge_index,
                _normalize_zero(ax),
                _normalize_zero(ay),
                _normalize_zero(az),
                _normalize_zero(bx),
                _normalize_zero(by),
                _normalize_zero(bz),
            )
        )
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# FRVWIRPCD1 wire point-content digest
# ---------------------------------------------------------------------------

_FRV_WIRE_PCD_FIELDS = (
    ("original_index", "<i8", 1),
    ("provider_state", "|u1", 1),
    ("relation_state", "|u1", 1),
    ("nearest_wire_id", "<i4", 1),
    ("nearest_segment_index", "<i4", 1),
    ("nearest_edge_index", "<i4", 1),
    ("centerline_distance_m", "<f8", 1),
    ("reason_flags", "<u4", 1),
    ("quality_flags", "<u4", 1),
)


def frv_wire_point_content_digest(
    *,
    source_content_identity_digest: str,
    source_identity_record_digest: str,
    authority_manifest_digest: str,
    wire_geometry_content_digest: str,
    provider_config_digest: str,
    source_point_count: int,
    chunks: Iterable[FrvWireChunk],
) -> str:
    """Return the FRVWIRPCD1 digest over the complete 0..N-1 wire domain.

    The framing is ASCII ``FRVWIRPCD1\\0``, five NUL-terminated SHA-256 headers
    (source content, source record, authority manifest, wire geometry content,
    provider config), ``uint64 N``, then per ndarray field a ``uint16``
    field-name byte length, the UTF-8 field name, a ``uint32`` single-row byte
    length, and the little-endian C-order row bytes in original_index order.
    """

    _require_sha(source_content_identity_digest, "source_content_identity_digest")
    _require_sha(source_identity_record_digest, "source_identity_record_digest")
    _require_sha(authority_manifest_digest, "authority_manifest_digest")
    _require_sha(wire_geometry_content_digest, "wire_geometry_content_digest")
    _require_sha(provider_config_digest, "provider_config_digest")
    source_point_count = _require_nonnegative_int(source_point_count, "source_point_count")

    hasher = hashlib.sha256()
    hasher.update(b"FRVWIRPCD1\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_identity_record_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(authority_manifest_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(wire_geometry_content_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(provider_config_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", source_point_count))
    expected_index = 0
    for chunk in chunks:
        if type(chunk) is not FrvWireChunk:
            raise TypeError("chunks must contain FrvWireChunk values")
        for row in range(chunk.point_count):
            index = int(chunk.original_index[row])
            if index != expected_index:
                raise ValueError("chunks must cover the complete 0..N-1 domain")
            expected_index += 1
    if expected_index != source_point_count:
        raise ValueError("chunks must cover the complete 0..N-1 domain")

    for name, dtype_str, count in _FRV_WIRE_PCD_FIELDS:
        name_bytes = name.encode("utf-8")
        if len(name_bytes) > 0xFFFF:
            raise ValueError("field name too long for the FRVWIRPCD1 framing")
        dtype = np.dtype(dtype_str)
        row_byte_length = dtype.itemsize * count
        if row_byte_length > 0xFFFFFFFF:
            raise ValueError("row byte length too long for the FRVWIRPCD1 framing")
        hasher.update(struct.pack("<H", len(name_bytes)))
        hasher.update(name_bytes)
        hasher.update(struct.pack("<I", row_byte_length))
        for chunk in chunks:
            array = getattr(chunk, name)
            for row in range(chunk.point_count):
                if dtype_str == "<f8":
                    row_bytes = (array[row] + np.zeros(1, dtype=dtype)).tobytes()
                else:
                    row_bytes = array[row].tobytes()
                hasher.update(row_bytes)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# FrvWireProviderProof
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FrvWireProviderProof:
    """Frozen, serializable exact-schema P-WIRE success proof."""

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
    authority_mode: FrvWireAuthorityMode
    authority_manifest_digest: str
    wire_geometry_content_digest: str
    wire_point_content_digest: str
    approved_wire_count: int
    wire_relation_valid_count: int
    wire_not_applicable_proven_count: int
    outside_influence_domain_count: int
    wire_body_possible_count: int
    wire_proximity_only_count: int
    wire_vegetation_conflict_count: int
    minimum_centerline_distance_m: float
    maximum_centerline_distance_m: float
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
        if self.schema_version != FRV_WIRE_PROOF_SCHEMA_VERSION:
            raise ValueError("unsupported P-WIRE proof schema version")
        _require_exact_str(self.provider_key, "provider_key")
        if self.provider_key != FRV_WIRE_PROVIDER_KEY:
            raise ValueError("provider_key must be wire_provider")
        _require_enum(self.provider_role, FrvA2ProviderRole, "provider_role")
        if self.provider_role is not FrvA2ProviderRole.WIRE:
            raise ValueError("provider_role must be WIRE")
        _require_exact_str(self.interface_version, "interface_version")
        if self.interface_version != FRV_WIRE_PROVIDER_INTERFACE_VERSION:
            raise ValueError("interface_version must match the frozen interface version")
        _require_exact_str(self.algorithm_version, "algorithm_version")
        if self.algorithm_version != FRV_WIRE_ALGORITHM_VERSION:
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
        _require_enum(self.authority_mode, FrvWireAuthorityMode, "authority_mode")
        _require_sha(self.authority_manifest_digest, "authority_manifest_digest")
        _require_sha(
            self.wire_geometry_content_digest, "wire_geometry_content_digest"
        )
        _require_sha(self.wire_point_content_digest, "wire_point_content_digest")
        for name in (
            "approved_wire_count",
            "wire_relation_valid_count",
            "wire_not_applicable_proven_count",
            "outside_influence_domain_count",
            "wire_body_possible_count",
            "wire_proximity_only_count",
            "wire_vegetation_conflict_count",
        ):
            object.__setattr__(
                self, name, _require_nonnegative_int(getattr(self, name), name)
            )
        object.__setattr__(
            self,
            "minimum_centerline_distance_m",
            _require_finite_float(
                self.minimum_centerline_distance_m, "minimum_centerline_distance_m"
            ),
        )
        object.__setattr__(
            self,
            "maximum_centerline_distance_m",
            _require_finite_float(
                self.maximum_centerline_distance_m, "maximum_centerline_distance_m"
            ),
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
                "proof_digest does not match the canonical P-WIRE proof fields"
            )

    def to_digest_dict(self) -> dict[str, Any]:
        return {
            "digest_domain": FRV_WIRE_PROOF_DIGEST_DOMAIN,
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
            "authority_mode": self.authority_mode.value,
            "authority_manifest_digest": self.authority_manifest_digest,
            "wire_geometry_content_digest": self.wire_geometry_content_digest,
            "wire_point_content_digest": self.wire_point_content_digest,
            "approved_wire_count": self.approved_wire_count,
            "wire_relation_valid_count": self.wire_relation_valid_count,
            "wire_not_applicable_proven_count": self.wire_not_applicable_proven_count,
            "outside_influence_domain_count": self.outside_influence_domain_count,
            "wire_body_possible_count": self.wire_body_possible_count,
            "wire_proximity_only_count": self.wire_proximity_only_count,
            "wire_vegetation_conflict_count": self.wire_vegetation_conflict_count,
            "minimum_centerline_distance_m": self.minimum_centerline_distance_m,
            "maximum_centerline_distance_m": self.maximum_centerline_distance_m,
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
    def from_dict(cls, data: Mapping[str, Any]) -> FrvWireProviderProof:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_role"] = _enum_from_value(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        values["domain_kind"] = _enum_from_value(
            values["domain_kind"], FrvA2DomainKind, "domain_kind"
        )
        values["authority_mode"] = _enum_from_value(
            values["authority_mode"], FrvWireAuthorityMode, "authority_mode"
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


def _validate_proof_structure(proof: FrvWireProviderProof) -> None:
    if type(proof) is not FrvWireProviderProof:
        raise ValueError("proof must be a FrvWireProviderProof")
    if proof.schema_version != FRV_WIRE_PROOF_SCHEMA_VERSION:
        raise ValueError("schema_version must match the frozen proof schema version")
    if proof.provider_key != FRV_WIRE_PROVIDER_KEY:
        raise ValueError("provider_key must be wire_provider")
    if type(proof.provider_role) is not FrvA2ProviderRole:
        raise ValueError("provider_role must be a FrvA2ProviderRole")
    if proof.provider_role is not FrvA2ProviderRole.WIRE:
        raise ValueError("provider_role must be WIRE")
    if proof.interface_version != FRV_WIRE_PROVIDER_INTERFACE_VERSION:
        raise ValueError("interface_version must match the frozen interface version")
    if proof.algorithm_version != FRV_WIRE_ALGORITHM_VERSION:
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
    if type(proof.authority_mode) is not FrvWireAuthorityMode:
        raise ValueError("authority_mode must be a FrvWireAuthorityMode")
    _require_sha(proof.authority_manifest_digest, "authority_manifest_digest")
    _require_sha(proof.wire_geometry_content_digest, "wire_geometry_content_digest")
    _require_sha(proof.wire_point_content_digest, "wire_point_content_digest")
    for name in (
        "approved_wire_count",
        "wire_relation_valid_count",
        "wire_not_applicable_proven_count",
        "outside_influence_domain_count",
        "wire_body_possible_count",
        "wire_proximity_only_count",
        "wire_vegetation_conflict_count",
    ):
        if type(getattr(proof, name)) is not int or getattr(proof, name) < 0:
            raise ValueError(f"{name} must be a nonnegative built-in int")
    _validate_vec3(proof.origin_xyz, "origin_xyz")
    _require_finite_float(
        proof.minimum_centerline_distance_m, "minimum_centerline_distance_m"
    )
    _require_finite_float(
        proof.maximum_centerline_distance_m, "maximum_centerline_distance_m"
    )
    if proof.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
        raise ValueError("coordinate_frame must be LOCAL_CARTESIAN_XYZ")
    if proof.units != "m":
        raise ValueError("units must be m")
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
        raise ValueError("proof_digest does not match the canonical P-WIRE proof fields")


def validate_frv_wire_provider_proof(proof: FrvWireProviderProof) -> None:
    _validate_proof_structure(proof)
    state_sum = (
        proof.wire_relation_valid_count + proof.wire_not_applicable_proven_count
    )
    if state_sum != proof.source_point_count:
        raise ValueError("provider state counts must sum to source_point_count")
    relation_sum = (
        proof.outside_influence_domain_count
        + proof.wire_body_possible_count
        + proof.wire_proximity_only_count
        + proof.wire_vegetation_conflict_count
    )
    if relation_sum != proof.source_point_count:
        raise ValueError("relation counts must sum to source_point_count")
    if proof.authority_mode is FrvWireAuthorityMode.EMPTY_INVENTORY:
        if proof.approved_wire_count != 0:
            raise ValueError("EMPTY authority requires zero approved wires")
        if proof.wire_relation_valid_count != 0:
            raise ValueError("EMPTY authority cannot have relation-valid rows")
        if proof.outside_influence_domain_count != proof.source_point_count:
            raise ValueError("EMPTY authority requires every row outside influence domain")
        if (
            proof.wire_body_possible_count != 0
            or proof.wire_proximity_only_count != 0
            or proof.wire_vegetation_conflict_count != 0
        ):
            raise ValueError("EMPTY authority cannot have in-domain relation counts")
        if (
            proof.minimum_centerline_distance_m != 0.0
            or proof.maximum_centerline_distance_m != 0.0
        ):
            raise ValueError("EMPTY authority distance aggregates must be positive zero")
    else:
        if proof.approved_wire_count <= 0:
            raise ValueError("W1 authority requires a positive approved wire count")
        if proof.wire_not_applicable_proven_count != 0:
            raise ValueError("non-empty authority cannot have not-applicable rows")
        if (
            proof.minimum_centerline_distance_m < 0.0
            or proof.maximum_centerline_distance_m < 0.0
            or proof.minimum_centerline_distance_m > proof.maximum_centerline_distance_m
        ):
            raise ValueError("non-empty distance aggregates must be an ordered finite range")
        in_domain = (
            proof.wire_body_possible_count
            + proof.wire_proximity_only_count
            + proof.wire_vegetation_conflict_count
        )
        if in_domain == 0:
            if (
                proof.minimum_centerline_distance_m != 0.0
                or proof.maximum_centerline_distance_m != 0.0
            ):
                raise ValueError(
                    "no in-domain point requires positive-zero distance aggregates"
                )
    expected_index_digest = _stream_index_domain_digest(
        proof.domain_count, proof.source_content_identity_digest
    )
    if proof.original_index_domain_digest != expected_index_digest:
        raise ValueError("original_index_domain_digest does not match the FULL_SOURCE domain")
    if proof.proof_digest != proof.computed_digest():
        raise ValueError("proof_digest does not match the canonical P-WIRE proof fields")


# ---------------------------------------------------------------------------
# Finalized P-WIRE result with private capability
# ---------------------------------------------------------------------------

_MISSING = object()


class _FRV_WIRE_FINALIZED_CAPABILITY:
    """Module-private identity-only finalized P-WIRE capability token."""

    __slots__ = ()


@dataclass(frozen=True, slots=True, init=False)
class FrvWireProviderResult:
    """Non-serializable finalized P-WIRE success wrapper (private capability)."""

    proof: FrvWireProviderProof
    source_identity: SourceIdentity
    _finalized_capability: Any = field(init=False, repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrvWireProviderResult has no public constructor")


def _validate_finalized_wire_bindings(
    proof: FrvWireProviderProof, source_identity: SourceIdentity
) -> None:
    validate_frv_wire_provider_proof(proof)
    validate_source_identity_record_digest(source_identity)
    validate_source_content_identity_digest(source_identity)
    if proof.source_content_identity_digest != source_identity.computed_content_digest():
        raise ValueError("source_content_identity_digest must match the computed source identity")
    if proof.source_identity_record_digest != source_identity.computed_record_digest():
        raise ValueError("source_identity_record_digest must match the computed source identity")
    if len(proof.artifact_components) < 1:
        raise ValueError("at least one artifact component is required")


def _create_finalized_frv_wire_provider_result(
    proof: FrvWireProviderProof,
    source_identity: SourceIdentity,
    *,
    capability: Any,
) -> FrvWireProviderResult:
    if type(proof) is not FrvWireProviderProof:
        raise TypeError("proof must be a FrvWireProviderProof")
    if type(source_identity) is not SourceIdentity:
        raise TypeError("source_identity must be a SourceIdentity")
    if capability is not _FRV_WIRE_FINALIZED_CAPABILITY:
        raise ValueError("capability must be the module-private finalized capability")
    _validate_finalized_wire_bindings(proof, source_identity)
    result = object.__new__(FrvWireProviderResult)
    object.__setattr__(result, "proof", proof)
    object.__setattr__(result, "source_identity", source_identity)
    object.__setattr__(result, "_finalized_capability", capability)
    return result


def validate_finalized_frv_wire_provider_result(result: FrvWireProviderResult) -> None:
    if type(result) is not FrvWireProviderResult:
        raise TypeError("result must be a FrvWireProviderResult")
    if getattr(result, "_finalized_capability", _MISSING) is not (
        _FRV_WIRE_FINALIZED_CAPABILITY
    ):
        raise ValueError("result must carry the module-private finalized capability")
    proof = getattr(result, "proof", _MISSING)
    if type(proof) is not FrvWireProviderProof:
        raise ValueError("result.proof must be a FrvWireProviderProof")
    source_identity = getattr(result, "source_identity", _MISSING)
    if type(source_identity) is not SourceIdentity:
        raise ValueError("result.source_identity must be a SourceIdentity")
    _validate_finalized_wire_bindings(proof, source_identity)


# ---------------------------------------------------------------------------
# FrvWireLoadError
# ---------------------------------------------------------------------------


class FrvWireLoadError(Exception):
    """P-WIRE load failure carrying a frozen A2 failure record."""

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
# FrvWireLoadSession protocol
# ---------------------------------------------------------------------------


class FrvWireLoadSession(Protocol):
    """Single-use P-WIRE load session interface (contracts only)."""

    def iter_chunks(self) -> Iterator[FrvWireChunk]: ...
    def finalize(self) -> FrvWireProviderResult: ...
    def close(self) -> None: ...
    def __enter__(self) -> FrvWireLoadSession: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]: ...


# ---------------------------------------------------------------------------
# FRVA2BND1 adapter
# ---------------------------------------------------------------------------


def to_frv_a2_wire_provider_binding_evidence(
    result: FrvWireProviderResult,
) -> FrvA2ProviderBindingEvidence:
    """Return the success-only FRVA2BND1 formal provider evidence for P-WIRE."""

    validate_finalized_frv_wire_provider_result(result)
    proof = result.proof
    evidence = FrvA2ProviderBindingEvidence.__new__(FrvA2ProviderBindingEvidence)
    object.__setattr__(evidence, "provider_key", FRV_WIRE_PROVIDER_KEY)
    object.__setattr__(evidence, "provider_role", FrvA2ProviderRole.WIRE)
    object.__setattr__(evidence, "evidence_class", FrvA2EvidenceClass.FORMAL_PROVIDER)
    object.__setattr__(evidence, "public_module", "algorithms.vegetation.wire_provider")
    object.__setattr__(evidence, "public_type_name", "FrvWireProviderResult")
    object.__setattr__(evidence, "public_loader_name", "load_frv_wire")
    object.__setattr__(evidence, "interface_version", FRV_WIRE_PROVIDER_INTERFACE_VERSION)
    object.__setattr__(evidence, "schema_version", FRV_WIRE_PROVIDER_SCHEMA_VERSION)
    object.__setattr__(evidence, "algorithm_version", FRV_WIRE_ALGORITHM_VERSION)
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
                proof.authority_manifest_digest,
                proof.wire_geometry_content_digest,
                proof.wire_point_content_digest,
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
    "FRV_WIRE_PROVIDER_KEY",
    "FRV_WIRE_PROVIDER_INTERFACE_VERSION",
    "FRV_WIRE_PROVIDER_SCHEMA_VERSION",
    "FRV_WIRE_CONFIG_SCHEMA_VERSION",
    "FRV_WIRE_PROOF_SCHEMA_VERSION",
    "FRV_WIRE_ALGORITHM_VERSION",
    "FRV_WIRE_CONFIG_DIGEST_DOMAIN",
    "FRV_WIRE_POINT_CONTENT_DIGEST_DOMAIN",
    "FRV_WIRE_PROOF_DIGEST_DOMAIN",
    "FRV_WIRE_AUTHORITY_SCHEMA_VERSION",
    "FRV_WIRE_AUTHORITY_DIGEST_DOMAIN",
    "FRV_WIRE_GEOMETRY_DIGEST_DOMAIN",
    "FRV_WIRE_DEFAULT_BODY_POSSIBLE_RADIUS_M",
    "FRV_WIRE_DEFAULT_CONFLICT_RADIUS_M",
    "FRV_WIRE_DEFAULT_PROXIMITY_RADIUS_M",
    "FRV_WIRE_DEFAULT_SPATIAL_INDEX_CELL_SIZE_M",
    "FRV_WIRE_DEFAULT_MAX_EDGES_PER_INDEX_CELL",
    "FrvWireAuthorityMode",
    "FrvWirePointRelationState",
    "FrvWireAuthorityManifest",
    "FrvWireAuthorityReference",
    "FrvWireProviderConfig",
    "FrvWireChunk",
    "FrvWireProviderProof",
    "FrvWireProviderResult",
    "FrvWireLoadError",
    "FrvWireLoadSession",
    "validate_frv_wire_authority_manifest",
    "validate_frv_wire_provider_config",
    "validate_frv_wire_chunk",
    "validate_frv_wire_provider_proof",
    "validate_finalized_frv_wire_provider_result",
    "derive_frv_wire_provider_config_digest",
    "derive_frv_wire_authority_approval_digest",
    "derive_frv_wire_authority_manifest_digest",
    "frv_wire_geometry_content_digest",
    "frv_wire_point_content_digest",
    "to_frv_a2_wire_provider_binding_evidence",
]
