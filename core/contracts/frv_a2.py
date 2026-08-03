"""FRV-A2 immutable readiness, failure, and digest contracts.

This module contains the serializable part of FRV-A2 only.  It deliberately
does not load providers, inspect files, or expose a runtime handoff bundle.
The contracts use the FRV-A1 canonical JSON and digest primitives, while
keeping the three A2 digest domains explicit and acyclic.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from core.contracts.frv import (
    FrvConfiguration,
    FrvConfigurationPurpose,
    FrvErrorCode,
    FrvReasonFlag,
    frv_digest,
    validate_configuration,
)

_A2_READINESS_SCHEMA_VERSION = "frv-a2-readiness-v1"
_A2_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_A2_INDEX_ROLES = frozenset({"ORIGINAL_INDEX", "SAMPLE_INDEX"})
_A2_LEGACY_PAIR_KEYS = frozenset({"a2a", "a2b", "v1", "v2", "v2a", "w1"})
_A2_SUPPORTING_MAPPINGS = {
    "a2b": ("hag", "supporting_evidence", "supporting_only"),
    "a2a": ("geometry", "supporting_evidence", "supporting_only"),
    "w1": ("wire", "supporting_evidence", "supporting_only"),
    "i1": ("provenance", "supporting_evidence", "supporting_only"),
}


class FrvA2ProviderRole(str, Enum):
    SOURCE_SCAN = "source_scan"
    HAG = "hag"
    GEOMETRY = "geometry"
    WIRE = "wire"
    PROVENANCE = "provenance"


_A2_BASE_REQUIRED_ROLES = (
    FrvA2ProviderRole.SOURCE_SCAN,
    FrvA2ProviderRole.HAG,
    FrvA2ProviderRole.GEOMETRY,
)


class FrvA2EvidenceClass(str, Enum):
    FORMAL_PROVIDER = "formal_provider"
    SUPPORTING_EVIDENCE = "supporting_evidence"


class FrvA2ProviderReadiness(str, Enum):
    VALIDATED = "validated"
    REVIEW_REQUIRED = "review_required"
    BLOCKED = "blocked"
    SUPPORTING_ONLY = "supporting_only"


class FrvA2AggregateReadiness(str, Enum):
    READY_FOR_A3 = "ready_for_a3"
    READY_FOR_A3_WITH_LIMITATIONS = "ready_for_a3_with_limitations"
    BLOCKED = "blocked"


class FrvA2DomainKind(str, Enum):
    FULL_SOURCE = "full_source"
    DECLARED_SUBSET = "declared_subset"
    SAMPLED_SUBSET = "sampled_subset"
    CANDIDATE_SUBSET = "candidate_subset"
    CLUSTER_SUBSET = "cluster_subset"
    PAIR_SUBSET = "pair_subset"
    ARTIFACT_REFERENCE = "artifact_reference"
    INFLUENCE_DOMAIN = "influence_domain"


class FrvA2FailurePhase(str, Enum):
    STRUCTURE_INTEGRITY = "structure_integrity"
    IDENTITY_CONTEXT = "identity_context"
    DOMAIN_ALIGNMENT = "domain_alignment"
    SEMANTIC_READINESS = "semantic_readiness"
    CROSS_PROVIDER = "cross_provider"
    PRODUCTION_POLICY = "production_policy"


class FrvA2ArtifactComponentKind(str, Enum):
    BUNDLE_MANIFEST_JSON = "bundle_manifest_json"
    MANIFEST_JSON = "manifest_json"
    PAYLOAD_NPZ = "payload_npz"
    SOURCE_FILE = "source_file"
    REFERENCED_MANIFEST_JSON = "referenced_manifest_json"
    REFERENCED_PAYLOAD_NPZ = "referenced_payload_npz"


class FrvA2ReviewCode(str, Enum):
    FORMAL_PROVIDER_REVIEW_REQUIRED = "formal_provider_review_required"
    CALIBRATION_WIRE_CONFIG_UNBOUND = "calibration_wire_config_unbound"


class FrvA2LimitationCode(str, Enum):
    NONPRODUCTION_CONFIGURATION = "nonproduction_configuration"
    FORMAL_COMPLETENESS_FORBIDDEN = "formal_completeness_forbidden"
    MEMBERSHIP_PUBLICATION_FORBIDDEN = "membership_publication_forbidden"
    OPTIONAL_TEST_WIRE_PROVIDER_ABSENT = "optional_test_wire_provider_absent"
    FORMAL_PROVIDER_REVIEW_REQUIRED = "formal_provider_review_required"
    CALIBRATION_WIRE_CONFIG_UNBOUND = "calibration_wire_config_unbound"


def _require_exact_str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise ValueError(f"{name} must be a built-in str")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be non-empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    return value


def _require_stable_text(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if any(character in text for character in ("/", "\\", ":", "@")):
        raise ValueError(f"{name} must not contain a path or host delimiter")
    if any(character.isspace() for character in text):
        raise ValueError(f"{name} must not contain whitespace")
    return text


def _require_key(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if _A2_KEY_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{name} must match the FRV-A2 key grammar")
    return text


def _require_sha(value: Any, name: str) -> str:
    text = _require_exact_str(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return text


def _require_sha_or_none(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha(value, name)


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


def _enum_from_json(value: Any, enum_type: type[Enum], name: str) -> Enum:
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
    value: Any, enum_type: type[Enum], name: str, *, declaration_order: bool = True
) -> tuple[Enum, ...]:
    items = _require_tuple(value, name)
    order = _enum_order(enum_type)
    seen: set[Enum] = set()
    for item in items:
        _require_enum(item, enum_type, f"{name}[]")
        if item in seen:
            raise ValueError(f"{name} must be unique")
        seen.add(item)
    if declaration_order and tuple(sorted(items, key=order.__getitem__)) != items:
        raise ValueError(f"{name} must use declaration order")
    return items


def _validate_sha_tuple(value: Any, name: str) -> tuple[str, ...]:
    items = _require_tuple(value, name)
    for item in items:
        _require_sha(item, f"{name}[]")
    if len(set(items)) != len(items) or tuple(sorted(items)) != items:
        raise ValueError(f"{name} must be sorted and unique")
    return items


def _component_sort_key(component: FrvA2ArtifactComponentBinding) -> tuple[int, str]:
    return (
        _enum_order(FrvA2ArtifactComponentKind)[component.component_kind],
        component.component_key,
    )


def _validate_component_tuple(
    value: Any, name: str
) -> tuple[FrvA2ArtifactComponentBinding, ...]:
    items = _require_tuple(value, name)
    if not items:
        raise ValueError(f"{name} must contain at least one component")
    for item in items:
        validate_frv_a2_artifact_component_binding(item)
    if len({item.component_key for item in items}) != len(items):
        raise ValueError(f"{name} must have unique component_key values")
    if tuple(sorted(items, key=_component_sort_key)) != items:
        raise ValueError(f"{name} must use canonical component order")
    return items


def _derive_artifact_reference_id(
    provider_role: FrvA2ProviderRole,
    provider_key: str,
    components: tuple[FrvA2ArtifactComponentBinding, ...],
) -> str:
    sequence = ";".join(
        ",".join(
            (
                component.component_kind.value,
                component.component_key,
                component.sha256,
                str(component.size_bytes),
            )
        )
        for component in components
    )
    return f"frv-a2-ref-v1|{provider_role.value}|{provider_key}|{sequence}"


def _legacy_pair_layout(
    components: tuple[FrvA2ArtifactComponentBinding, ...], name: str
) -> None:
    expected = {
        "manifest_json": FrvA2ArtifactComponentKind.MANIFEST_JSON,
        "payload_npz": FrvA2ArtifactComponentKind.PAYLOAD_NPZ,
    }
    actual = {component.component_key: component.component_kind for component in components}
    if actual != expected or len(components) != len(expected):
        raise ValueError(f"{name} must be the exact manifest_json/payload_npz pair")


def _i1_layout(components: tuple[FrvA2ArtifactComponentBinding, ...], name: str) -> None:
    required = {
        "bundle_manifest_json": FrvA2ArtifactComponentKind.BUNDLE_MANIFEST_JSON,
        "w1_manifest_json": FrvA2ArtifactComponentKind.REFERENCED_MANIFEST_JSON,
        "w1_payload_npz": FrvA2ArtifactComponentKind.REFERENCED_PAYLOAD_NPZ,
        "v2_manifest_json": FrvA2ArtifactComponentKind.REFERENCED_MANIFEST_JSON,
        "v2_payload_npz": FrvA2ArtifactComponentKind.REFERENCED_PAYLOAD_NPZ,
    }
    optional = {
        "v2a_manifest_json": FrvA2ArtifactComponentKind.REFERENCED_MANIFEST_JSON,
        "v2a_payload_npz": FrvA2ArtifactComponentKind.REFERENCED_PAYLOAD_NPZ,
    }
    actual = {component.component_key: component.component_kind for component in components}
    if not set(required).issubset(actual):
        raise ValueError(f"{name} is missing mandatory I1 components")
    if set(actual) - set(required) - set(optional):
        raise ValueError(f"{name} contains unknown I1 components")
    if set(actual) & set(optional) not in (set(), set(optional)):
        raise ValueError(f"{name} requires an atomic V2A component pair")
    expected = {**required, **{key: optional[key] for key in actual if key in optional}}
    if actual != expected or len(actual) != len(expected):
        raise ValueError(f"{name} has an invalid I1 component kind")


def _validate_component_layout(
    provider_key: str,
    provider_role: FrvA2ProviderRole,
    components: tuple[FrvA2ArtifactComponentBinding, ...],
) -> None:
    if provider_key in _A2_LEGACY_PAIR_KEYS:
        _legacy_pair_layout(components, "artifact_components")
    if provider_key == "i1" or provider_role is FrvA2ProviderRole.PROVENANCE:
        _i1_layout(components, "artifact_components")


@dataclass(frozen=True)
class FrvA2FailureRecord:
    phase: FrvA2FailurePhase
    provider_role: FrvA2ProviderRole
    provider_key: str
    field_name: str
    error_code: FrvErrorCode
    reason_flags: int
    detail: str

    def __post_init__(self) -> None:
        _validate_failure(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "provider_role": self.provider_role.value,
            "provider_key": self.provider_key,
            "field_name": self.field_name,
            "error_code": self.error_code.value,
            "reason_flags": self.reason_flags,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrvA2FailureRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["phase"] = _enum_from_json(values["phase"], FrvA2FailurePhase, "phase")
        values["provider_role"] = _enum_from_json(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        values["error_code"] = _enum_from_json(values["error_code"], FrvErrorCode, "error_code")
        return cls(**values)


def _failure_key(record: FrvA2FailureRecord) -> tuple[int, int, str, str, str, int]:
    return (
        _enum_order(FrvA2FailurePhase)[record.phase],
        _enum_order(FrvA2ProviderRole)[record.provider_role],
        record.provider_key,
        record.field_name,
        record.error_code.value,
        record.reason_flags,
    )


def _validate_failure(record: FrvA2FailureRecord) -> None:
    if type(record) is not FrvA2FailureRecord:
        raise ValueError("record must be a FrvA2FailureRecord")
    _require_enum(record.phase, FrvA2FailurePhase, "phase")
    _require_enum(record.provider_role, FrvA2ProviderRole, "provider_role")
    _require_key(record.provider_key, "provider_key")
    _require_key(record.field_name, "field_name")
    _require_enum(record.error_code, FrvErrorCode, "error_code")
    _require_nonnegative_int(record.reason_flags, "reason_flags")
    expected = (
        f"{record.phase.value}:{record.provider_role.value}:{record.provider_key}:"
        f"{record.field_name}:{record.error_code.value}:{record.reason_flags}"
    )
    if type(record.detail) is not str or record.detail != expected:
        raise ValueError("detail must be the canonical derived failure detail")


def validate_frv_a2_failure_record(record: FrvA2FailureRecord) -> None:
    _validate_failure(record)


@dataclass(frozen=True)
class FrvA2ArtifactComponentBinding:
    component_key: str
    component_kind: FrvA2ArtifactComponentKind
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        _validate_component(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_key": self.component_key,
            "component_kind": self.component_kind.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrvA2ArtifactComponentBinding:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["component_kind"] = _enum_from_json(
            values["component_kind"], FrvA2ArtifactComponentKind, "component_kind"
        )
        return cls(**values)


def _validate_component(record: FrvA2ArtifactComponentBinding) -> None:
    if type(record) is not FrvA2ArtifactComponentBinding:
        raise ValueError("record must be a FrvA2ArtifactComponentBinding")
    _require_key(record.component_key, "component_key")
    _require_enum(record.component_kind, FrvA2ArtifactComponentKind, "component_kind")
    _require_sha(record.sha256, "sha256")
    _require_nonnegative_int(record.size_bytes, "size_bytes")


def validate_frv_a2_artifact_component_binding(
    record: FrvA2ArtifactComponentBinding,
) -> None:
    _validate_component(record)


def _provider_binding_fields(record: FrvA2ProviderBindingEvidence) -> dict[str, Any]:
    return {
        "provider_key": record.provider_key,
        "provider_role": record.provider_role.value,
        "evidence_class": record.evidence_class.value,
        "public_module": record.public_module,
        "public_type_name": record.public_type_name,
        "public_loader_name": record.public_loader_name,
        "interface_version": record.interface_version,
        "schema_version": record.schema_version,
        "algorithm_version": record.algorithm_version,
        "configuration_digest": record.configuration_digest,
        "provider_config_digest": record.provider_config_digest,
        "artifact_reference_id": record.artifact_reference_id,
        "artifact_components": [component.to_dict() for component in record.artifact_components],
        "source_content_identity_digest": record.source_content_identity_digest,
        "source_identity_record_digest": record.source_identity_record_digest,
        "coordinate_frame": record.coordinate_frame,
        "units": record.units,
        "origin_xyz": list(record.origin_xyz),
        "transform_declaration": record.transform_declaration,
        "domain_kind": record.domain_kind.value,
        "domain_count": record.domain_count,
        "original_index_domain_digest": record.original_index_domain_digest,
        "sample_index_domain_digest": record.sample_index_domain_digest,
        "provider_readiness": record.provider_readiness.value,
        "quality_flags": record.quality_flags,
        "reason_flags": record.reason_flags,
        "error_codes": [code.value for code in record.error_codes],
        "upstream_binding_digests": list(record.upstream_binding_digests),
        "review_codes": [code.value for code in record.review_codes],
        "limitation_codes": [code.value for code in record.limitation_codes],
    }


def _provider_binding_payload(record: FrvA2ProviderBindingEvidence) -> dict[str, Any]:
    return {"digest_domain": "FRVA2BND1", **_provider_binding_fields(record)}


@dataclass(frozen=True)
class FrvA2ProviderBindingEvidence:
    provider_key: str
    provider_role: FrvA2ProviderRole
    evidence_class: FrvA2EvidenceClass
    public_module: str
    public_type_name: str
    public_loader_name: str
    interface_version: str
    schema_version: str | None
    algorithm_version: str
    configuration_digest: str
    provider_config_digest: str | None
    artifact_reference_id: str
    artifact_components: tuple[FrvA2ArtifactComponentBinding, ...]
    source_content_identity_digest: str
    source_identity_record_digest: str
    coordinate_frame: str
    units: str
    origin_xyz: tuple[float, float, float]
    transform_declaration: str
    domain_kind: FrvA2DomainKind
    domain_count: int
    original_index_domain_digest: str | None
    sample_index_domain_digest: str | None
    provider_readiness: FrvA2ProviderReadiness
    quality_flags: int
    reason_flags: int
    error_codes: tuple[FrvErrorCode, ...]
    upstream_binding_digests: tuple[str, ...]
    review_codes: tuple[FrvA2ReviewCode, ...]
    limitation_codes: tuple[FrvA2LimitationCode, ...]
    binding_digest: str

    def __post_init__(self) -> None:
        _validate_provider(self)
        computed = frv_a2_provider_binding_digest(self)
        if self.binding_digest != computed:
            raise ValueError("binding_digest does not match canonical A2 fields")

    def to_dict(self) -> dict[str, Any]:
        result = _provider_binding_fields(self)
        result["binding_digest"] = self.binding_digest
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrvA2ProviderBindingEvidence:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_role"] = _enum_from_json(
            values["provider_role"], FrvA2ProviderRole, "provider_role"
        )
        values["evidence_class"] = _enum_from_json(
            values["evidence_class"], FrvA2EvidenceClass, "evidence_class"
        )
        values["domain_kind"] = _enum_from_json(
            values["domain_kind"], FrvA2DomainKind, "domain_kind"
        )
        values["provider_readiness"] = _enum_from_json(
            values["provider_readiness"], FrvA2ProviderReadiness, "provider_readiness"
        )
        values["artifact_components"] = tuple(
            FrvA2ArtifactComponentBinding.from_dict(item)
            for item in _tuple_from_json(values["artifact_components"], "artifact_components")
        )
        values["origin_xyz"] = tuple(_tuple_from_json(values["origin_xyz"], "origin_xyz"))
        values["error_codes"] = tuple(
            _enum_from_json(item, FrvErrorCode, "error_codes[]")
            for item in _tuple_from_json(values["error_codes"], "error_codes")
        )
        values["upstream_binding_digests"] = _tuple_from_json(
            values["upstream_binding_digests"], "upstream_binding_digests"
        )
        values["review_codes"] = tuple(
            _enum_from_json(item, FrvA2ReviewCode, "review_codes[]")
            for item in _tuple_from_json(values["review_codes"], "review_codes")
        )
        values["limitation_codes"] = tuple(
            _enum_from_json(item, FrvA2LimitationCode, "limitation_codes[]")
            for item in _tuple_from_json(values["limitation_codes"], "limitation_codes")
        )
        return cls(**values)


def _validate_provider(record: FrvA2ProviderBindingEvidence) -> None:
    if type(record) is not FrvA2ProviderBindingEvidence:
        raise ValueError("record must be a FrvA2ProviderBindingEvidence")
    _require_key(record.provider_key, "provider_key")
    _require_enum(record.provider_role, FrvA2ProviderRole, "provider_role")
    _require_enum(record.evidence_class, FrvA2EvidenceClass, "evidence_class")
    for name in (
        "public_module",
        "public_type_name",
        "public_loader_name",
        "interface_version",
        "algorithm_version",
        "coordinate_frame",
        "units",
        "transform_declaration",
    ):
        _require_stable_text(getattr(record, name), name)
    if record.schema_version is not None:
        _require_stable_text(record.schema_version, "schema_version")
    _require_sha(record.configuration_digest, "configuration_digest")
    _require_sha_or_none(record.provider_config_digest, "provider_config_digest")
    components = _validate_component_tuple(record.artifact_components, "artifact_components")
    _validate_component_layout(record.provider_key, record.provider_role, components)
    expected_reference = _derive_artifact_reference_id(
        record.provider_role, record.provider_key, components
    )
    if type(record.artifact_reference_id) is not str:
        raise ValueError("artifact_reference_id must be a built-in str")
    if record.artifact_reference_id != expected_reference:
        raise ValueError("artifact_reference_id does not match canonical components")
    _require_sha(record.source_content_identity_digest, "source_content_identity_digest")
    _require_sha(record.source_identity_record_digest, "source_identity_record_digest")
    origin = _require_tuple(record.origin_xyz, "origin_xyz")
    if len(origin) != 3:
        raise ValueError("origin_xyz must contain exactly three values")
    for index, value in enumerate(origin):
        _require_finite_float(value, f"origin_xyz[{index}]")
    _require_enum(record.domain_kind, FrvA2DomainKind, "domain_kind")
    _require_nonnegative_int(record.domain_count, "domain_count")
    _require_sha_or_none(record.original_index_domain_digest, "original_index_domain_digest")
    _require_sha_or_none(record.sample_index_domain_digest, "sample_index_domain_digest")
    _require_enum(record.provider_readiness, FrvA2ProviderReadiness, "provider_readiness")
    _require_nonnegative_int(record.quality_flags, "quality_flags")
    _require_nonnegative_int(record.reason_flags, "reason_flags")
    _validate_enum_tuple(record.error_codes, FrvErrorCode, "error_codes")
    _validate_sha_tuple(record.upstream_binding_digests, "upstream_binding_digests")
    _validate_enum_tuple(record.review_codes, FrvA2ReviewCode, "review_codes")
    _validate_enum_tuple(record.limitation_codes, FrvA2LimitationCode, "limitation_codes")
    _require_sha(record.binding_digest, "binding_digest")
    mapping = _A2_SUPPORTING_MAPPINGS.get(record.provider_key)
    if mapping is not None:
        expected_role, expected_class, expected_readiness = mapping
        if (
            record.evidence_class.value != expected_class
            or record.provider_role.value != expected_role
            or record.provider_readiness.value != expected_readiness
        ):
            raise ValueError("fixed legacy evidence mapping is supporting-only")
    if record.evidence_class is FrvA2EvidenceClass.SUPPORTING_EVIDENCE:
        if record.provider_readiness is not FrvA2ProviderReadiness.SUPPORTING_ONLY:
            raise ValueError("supporting evidence must be SUPPORTING_ONLY")
    else:
        if record.provider_readiness is FrvA2ProviderReadiness.SUPPORTING_ONLY:
            raise ValueError("formal providers cannot be SUPPORTING_ONLY")
        if record.provider_role is FrvA2ProviderRole.PROVENANCE:
            raise ValueError("PROVENANCE is reserved for supporting reference evidence")


def validate_frv_a2_provider_binding_evidence(
    record: FrvA2ProviderBindingEvidence,
) -> None:
    _validate_provider(record)
    computed = frv_a2_provider_binding_digest(record)
    if record.binding_digest != computed:
        raise ValueError("binding_digest does not match canonical A2 fields")


def frv_a2_provider_binding_digest(record: FrvA2ProviderBindingEvidence) -> str:
    """Return the FRVA2BND1 digest, excluding the stored self-digest."""

    if type(record) is not FrvA2ProviderBindingEvidence:
        raise ValueError("record must be a FrvA2ProviderBindingEvidence")
    return frv_digest(_provider_binding_payload(record))


def _readiness_fields(record: FrvA2ReadinessRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "source_content_identity_digest": record.source_content_identity_digest,
        "source_identity_record_digest": record.source_identity_record_digest,
        "configuration_digest": record.configuration_digest,
        "provider_evidence": [evidence.to_dict() for evidence in record.provider_evidence],
        "aggregate_readiness": record.aggregate_readiness.value,
        "blocking_failures": [failure.to_dict() for failure in record.blocking_failures],
        "review_codes": [code.value for code in record.review_codes],
        "limitation_codes": [code.value for code in record.limitation_codes],
    }


def _readiness_payload(record: FrvA2ReadinessRecord) -> dict[str, Any]:
    return {
        "digest_domain": "FRVA2RDY1",
        "schema_version": record.schema_version,
        "source_content_identity_digest": record.source_content_identity_digest,
        "source_identity_record_digest": record.source_identity_record_digest,
        "configuration_digest": record.configuration_digest,
        "provider_evidence_projections": [
            _provider_binding_payload(evidence) for evidence in record.provider_evidence
        ],
        "aggregate_readiness": record.aggregate_readiness.value,
        "blocking_failures": [failure.to_dict() for failure in record.blocking_failures],
        "review_codes": [code.value for code in record.review_codes],
        "limitation_codes": [code.value for code in record.limitation_codes],
    }


@dataclass(frozen=True)
class FrvA2ReadinessRecord:
    schema_version: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    configuration_digest: str
    provider_evidence: tuple[FrvA2ProviderBindingEvidence, ...]
    aggregate_readiness: FrvA2AggregateReadiness
    blocking_failures: tuple[FrvA2FailureRecord, ...]
    review_codes: tuple[FrvA2ReviewCode, ...]
    limitation_codes: tuple[FrvA2LimitationCode, ...]
    result_digest: str

    def __post_init__(self) -> None:
        _validate_readiness(self)
        computed = frv_a2_readiness_record_digest(self)
        if self.result_digest != computed:
            raise ValueError("result_digest does not match canonical A2 fields")

    def to_dict(self) -> dict[str, Any]:
        result = _readiness_fields(self)
        result["result_digest"] = self.result_digest
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrvA2ReadinessRecord:
        values = _exact_fields(data, set(cls.__dataclass_fields__), cls.__name__)
        values["provider_evidence"] = tuple(
            FrvA2ProviderBindingEvidence.from_dict(item)
            for item in _tuple_from_json(values["provider_evidence"], "provider_evidence")
        )
        values["aggregate_readiness"] = _enum_from_json(
            values["aggregate_readiness"], FrvA2AggregateReadiness, "aggregate_readiness"
        )
        values["blocking_failures"] = tuple(
            FrvA2FailureRecord.from_dict(item)
            for item in _tuple_from_json(values["blocking_failures"], "blocking_failures")
        )
        values["review_codes"] = tuple(
            _enum_from_json(item, FrvA2ReviewCode, "review_codes[]")
            for item in _tuple_from_json(values["review_codes"], "review_codes")
        )
        values["limitation_codes"] = tuple(
            _enum_from_json(item, FrvA2LimitationCode, "limitation_codes[]")
            for item in _tuple_from_json(values["limitation_codes"], "limitation_codes")
        )
        return cls(**values)


def _provider_sort_key(
    record: FrvA2ProviderBindingEvidence,
) -> tuple[int, str]:
    return (_enum_order(FrvA2ProviderRole)[record.provider_role], record.provider_key)


def _validate_base_formal_role_gate(record: FrvA2ReadinessRecord) -> None:
    if record.aggregate_readiness is FrvA2AggregateReadiness.BLOCKED:
        return
    formal_by_role: dict[FrvA2ProviderRole, list[FrvA2ProviderBindingEvidence]] = {}
    for evidence in record.provider_evidence:
        if evidence.evidence_class is FrvA2EvidenceClass.FORMAL_PROVIDER:
            formal_by_role.setdefault(evidence.provider_role, []).append(evidence)
    for role in _A2_BASE_REQUIRED_ROLES:
        candidates = formal_by_role.get(role, [])
        if len(candidates) != 1:
            raise ValueError(f"base formal role {role.value} is not uniquely bound")
        readiness = candidates[0].provider_readiness
        if readiness in {
            FrvA2ProviderReadiness.SUPPORTING_ONLY,
            FrvA2ProviderReadiness.BLOCKED,
        }:
            raise ValueError(f"base formal role {role.value} is not eligible")
        if (
            record.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3
            and readiness is not FrvA2ProviderReadiness.VALIDATED
        ):
            raise ValueError(f"base formal role {role.value} is not validated")
        if (
            record.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS
            and readiness
            not in {
                FrvA2ProviderReadiness.VALIDATED,
                FrvA2ProviderReadiness.REVIEW_REQUIRED,
            }
        ):
            raise ValueError(f"base formal role {role.value} is not reviewable")


def _validate_readiness(record: FrvA2ReadinessRecord) -> None:
    if type(record) is not FrvA2ReadinessRecord:
        raise ValueError("record must be a FrvA2ReadinessRecord")
    if (
        type(record.schema_version) is not str
        or record.schema_version != _A2_READINESS_SCHEMA_VERSION
    ):
        raise ValueError("schema_version must be frv-a2-readiness-v1")
    _require_sha(record.source_content_identity_digest, "source_content_identity_digest")
    _require_sha(record.source_identity_record_digest, "source_identity_record_digest")
    _require_sha(record.configuration_digest, "configuration_digest")
    _require_tuple(record.provider_evidence, "provider_evidence")
    for item in record.provider_evidence:
        validate_frv_a2_provider_binding_evidence(item)
    if tuple(sorted(record.provider_evidence, key=_provider_sort_key)) != record.provider_evidence:
        raise ValueError("provider_evidence must use canonical role/key order")
    provider_keys = [item.provider_key for item in record.provider_evidence]
    artifact_ids = [item.artifact_reference_id for item in record.provider_evidence]
    if len(set(provider_keys)) != len(provider_keys):
        raise ValueError("provider_key must be unique in a readiness record")
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("artifact_reference_id must be unique in a readiness record")
    _require_enum(record.aggregate_readiness, FrvA2AggregateReadiness, "aggregate_readiness")
    _validate_base_formal_role_gate(record)
    _require_tuple(record.blocking_failures, "blocking_failures")
    for failure in record.blocking_failures:
        validate_frv_a2_failure_record(failure)
    if tuple(sorted(record.blocking_failures, key=_failure_key)) != record.blocking_failures:
        raise ValueError("blocking_failures must use canonical total order")
    failure_keys = [_failure_key(failure) for failure in record.blocking_failures]
    if len(set(failure_keys)) != len(failure_keys):
        raise ValueError("blocking_failures must be deduplicated")
    _validate_enum_tuple(record.review_codes, FrvA2ReviewCode, "review_codes")
    _validate_enum_tuple(record.limitation_codes, FrvA2LimitationCode, "limitation_codes")
    if record.aggregate_readiness is FrvA2AggregateReadiness.BLOCKED:
        if not record.blocking_failures:
            raise ValueError("BLOCKED readiness requires a blocking failure")
    elif record.blocking_failures:
        raise ValueError("ready readiness cannot contain blocking failures")
    if record.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3:
        if record.review_codes or record.limitation_codes:
            raise ValueError("READY_FOR_A3 cannot contain review or limitation codes")
        if any(
            item.evidence_class is FrvA2EvidenceClass.FORMAL_PROVIDER
            and item.provider_readiness is not FrvA2ProviderReadiness.VALIDATED
            for item in record.provider_evidence
        ):
            raise ValueError("READY_FOR_A3 requires every included formal provider to be validated")
    elif (
        record.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS
        and not record.review_codes
        and not record.limitation_codes
    ):
        raise ValueError("READY_FOR_A3_WITH_LIMITATIONS requires a review or limitation code")
    _require_sha(record.result_digest, "result_digest")


def _validate_readiness_against_configuration(
    record: FrvA2ReadinessRecord, configuration: FrvConfiguration
) -> None:
    required_roles = _derive_required_formal_roles(configuration)
    half_bound_failure = _test_only_half_bound_failure(configuration)
    if half_bound_failure is not None:
        if record.aggregate_readiness is not FrvA2AggregateReadiness.BLOCKED:
            raise ValueError("TEST_ONLY half-bound wire configuration must be BLOCKED")
        if half_bound_failure not in record.blocking_failures:
            raise ValueError("TEST_ONLY half-bound wire failure is missing")
        return
    formal_by_role: dict[FrvA2ProviderRole, list[FrvA2ProviderBindingEvidence]] = {}
    for evidence in record.provider_evidence:
        if evidence.evidence_class is FrvA2EvidenceClass.FORMAL_PROVIDER:
            formal_by_role.setdefault(evidence.provider_role, []).append(evidence)
    if (
        configuration.configuration_purpose is FrvConfigurationPurpose.TEST_ONLY
        and configuration.wire_provider_interface_version is None
        and configuration.wire_provider_config_digest is None
        and formal_by_role.get(FrvA2ProviderRole.WIRE)
    ):
        raise ValueError("TEST_ONLY without wire binding cannot contain a formal wire provider")
    if record.aggregate_readiness is FrvA2AggregateReadiness.BLOCKED:
        return
    for role in required_roles:
        candidates = formal_by_role.get(role, [])
        if len(candidates) != 1:
            raise ValueError(f"required formal role {role.value} is not uniquely bound")
        evidence = candidates[0]
        if evidence.provider_readiness is FrvA2ProviderReadiness.BLOCKED:
            raise ValueError(f"required formal role {role.value} is blocked")
        if evidence.configuration_digest != configuration.configuration_digest:
            raise ValueError(f"required formal role {role.value} has a configuration mismatch")
        expected_provider_config_digest = {
            FrvA2ProviderRole.SOURCE_SCAN: configuration.scan_provider_config_digest,
            FrvA2ProviderRole.HAG: configuration.hag_provider_config_digest,
            FrvA2ProviderRole.GEOMETRY: configuration.geometry_provider_config_digest,
            FrvA2ProviderRole.WIRE: configuration.wire_provider_config_digest,
        }[role]
        if expected_provider_config_digest is not None and (
            evidence.provider_config_digest != expected_provider_config_digest
        ):
            raise ValueError(f"required formal role {role.value} has a provider config mismatch")
        if (
            record.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3
            and evidence.provider_readiness is not FrvA2ProviderReadiness.VALIDATED
        ):
            raise ValueError(f"required formal role {role.value} is not validated")
    if configuration.configuration_purpose in {
        FrvConfigurationPurpose.CALIBRATION_CANDIDATE,
        FrvConfigurationPurpose.TEST_ONLY,
    }:
        required_limitations = {
            FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION,
            FrvA2LimitationCode.FORMAL_COMPLETENESS_FORBIDDEN,
            FrvA2LimitationCode.MEMBERSHIP_PUBLICATION_FORBIDDEN,
        }
        if not required_limitations.issubset(set(record.limitation_codes)):
            raise ValueError("non-production configuration limitations are incomplete")
    if configuration.configuration_purpose is FrvConfigurationPurpose.CALIBRATION_CANDIDATE:
        wire = formal_by_role[FrvA2ProviderRole.WIRE][0]
        if configuration.wire_provider_config_digest is None:
            if wire.provider_readiness is not FrvA2ProviderReadiness.REVIEW_REQUIRED:
                raise ValueError("unbound calibration wire must require review")
            if (
                FrvA2ReviewCode.CALIBRATION_WIRE_CONFIG_UNBOUND not in record.review_codes
                or FrvA2LimitationCode.CALIBRATION_WIRE_CONFIG_UNBOUND
                not in record.limitation_codes
            ):
                raise ValueError("unbound calibration wire codes are incomplete")
    if configuration.configuration_purpose is FrvConfigurationPurpose.TEST_ONLY:
        wire_binding_is_complete = (
            configuration.wire_provider_interface_version is not None
            and configuration.wire_provider_config_digest is not None
        )
        optional_wire_absence = FrvA2LimitationCode.OPTIONAL_TEST_WIRE_PROVIDER_ABSENT
        if not wire_binding_is_complete:
            if optional_wire_absence not in record.limitation_codes:
                raise ValueError(
                    "TEST_ONLY without wire binding requires the optional-wire limitation"
                )
            if (
                record.aggregate_readiness
                is not FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS
            ):
                raise ValueError("TEST_ONLY without wire binding requires limited readiness")
        elif optional_wire_absence in record.limitation_codes:
            raise ValueError("TEST_ONLY with wire binding cannot claim optional-wire absence")


def validate_frv_a2_readiness_record(
    record: FrvA2ReadinessRecord,
    configuration: FrvConfiguration | None = None,
) -> None:
    _validate_readiness(record)
    if configuration is not None:
        if type(configuration) is not FrvConfiguration:
            raise ValueError("configuration must be a FrvConfiguration")
        validate_configuration(configuration)
        _validate_readiness_against_configuration(record, configuration)
    computed = frv_a2_readiness_record_digest(record)
    if record.result_digest != computed:
        raise ValueError("result_digest does not match canonical A2 fields")


def frv_a2_readiness_record_digest(record: FrvA2ReadinessRecord) -> str:
    """Return the FRVA2RDY1 digest from explicit provider projections."""

    if type(record) is not FrvA2ReadinessRecord:
        raise ValueError("record must be a FrvA2ReadinessRecord")
    return frv_digest(_readiness_payload(record))


def _index_value(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not (
        type(value) is int or isinstance(value, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0 or integer > 0xFFFFFFFFFFFFFFFF:
        raise ValueError(f"{name} must fit in uint64")
    return integer


def _iter_index_values(indices: Any) -> tuple[int, Any]:
    if type(indices) is tuple:
        return len(indices), (_index_value(value, "indices[]") for value in indices)
    if type(indices) is not np.ndarray:
        raise ValueError("indices must be a tuple or one-dimensional NumPy integer array")
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if indices.dtype.kind not in "iu":
        raise ValueError("indices must use a signed or unsigned integer dtype")

    def values() -> Any:
        for item in indices:
            integer = int(item)
            if integer < 0 or integer > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("indices[] must fit in uint64")
            yield integer

    return int(indices.shape[0]), values()


def frv_a2_index_domain_digest(
    index_role: str,
    source_content_identity_digest: str,
    domain_kind: FrvA2DomainKind,
    indices: tuple[int, ...] | np.ndarray,
) -> str:
    """Hash the declared index domain using the FRVA2IDX1 byte stream."""

    role = _require_exact_str(index_role, "index_role")
    if role not in _A2_INDEX_ROLES:
        raise ValueError("index_role must be ORIGINAL_INDEX or SAMPLE_INDEX")
    source_digest = _require_sha(source_content_identity_digest, "source_content_identity_digest")
    _require_enum(domain_kind, FrvA2DomainKind, "domain_kind")
    count, iterator = _iter_index_values(indices)
    hasher = hashlib.sha256()
    hasher.update(b"FRVA2IDX1\0")
    hasher.update(role.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(source_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(domain_kind.value.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", count))
    for value in iterator:
        hasher.update(struct.pack("<Q", value))
    return hasher.hexdigest()


def _derive_required_formal_roles(config: FrvConfiguration) -> tuple[FrvA2ProviderRole, ...]:
    """Derive A2 required roles privately from the already frozen A1 config."""

    if type(config) is not FrvConfiguration:
        raise ValueError("configuration must be a FrvConfiguration")
    validate_configuration(config)
    roles = [FrvA2ProviderRole.SOURCE_SCAN, FrvA2ProviderRole.HAG, FrvA2ProviderRole.GEOMETRY]
    purpose = config.configuration_purpose
    if purpose in {
        FrvConfigurationPurpose.PRODUCTION_APPROVED,
        FrvConfigurationPurpose.CALIBRATION_CANDIDATE,
    }:
        roles.append(FrvA2ProviderRole.WIRE)
    elif (
        config.wire_provider_interface_version is not None
        and config.wire_provider_config_digest is not None
    ):
        roles.append(FrvA2ProviderRole.WIRE)
    return tuple(roles)


def _test_only_half_bound_failure(config: FrvConfiguration) -> FrvA2FailureRecord | None:
    """Return the frozen fail-closed A2 record for TEST_ONLY half-binding."""

    _derive_required_formal_roles(config)
    if config.configuration_purpose is not FrvConfigurationPurpose.TEST_ONLY:
        return None
    interface_bound = config.wire_provider_interface_version is not None
    digest_bound = config.wire_provider_config_digest is not None
    if interface_bound == digest_bound:
        return None
    phase = FrvA2FailurePhase.PRODUCTION_POLICY
    role = FrvA2ProviderRole.WIRE
    return FrvA2FailureRecord(
        phase=phase,
        provider_role=role,
        provider_key="wire",
        field_name="wire_provider_binding",
        error_code=FrvErrorCode.FRV_PROVIDER_INVALID,
        reason_flags=int(FrvReasonFlag.PROVIDER_UNCERTAINTY),
        detail=(
            f"{phase.value}:{role.value}:wire:wire_provider_binding:"
            f"{FrvErrorCode.FRV_PROVIDER_INVALID.value}:{int(FrvReasonFlag.PROVIDER_UNCERTAINTY)}"
        ),
    )


__all__ = [
    "FrvA2ProviderRole",
    "FrvA2EvidenceClass",
    "FrvA2ProviderReadiness",
    "FrvA2AggregateReadiness",
    "FrvA2DomainKind",
    "FrvA2FailurePhase",
    "FrvA2ArtifactComponentKind",
    "FrvA2ReviewCode",
    "FrvA2LimitationCode",
    "FrvA2FailureRecord",
    "FrvA2ArtifactComponentBinding",
    "FrvA2ProviderBindingEvidence",
    "FrvA2ReadinessRecord",
    "frv_a2_index_domain_digest",
    "frv_a2_provider_binding_digest",
    "frv_a2_readiness_record_digest",
    "validate_frv_a2_failure_record",
    "validate_frv_a2_artifact_component_binding",
    "validate_frv_a2_provider_binding_evidence",
    "validate_frv_a2_readiness_record",
]
