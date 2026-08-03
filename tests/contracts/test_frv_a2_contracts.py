"""Focused FRV-A2 contract and synthetic-provider tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest
from core.contracts.frv import (
    FrvConfiguration,
    FrvErrorCode,
    FrvReasonFlag,
    frv_digest,
)
from core.contracts.frv_a2 import (
    FrvA2AggregateReadiness,
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2DomainKind,
    FrvA2EvidenceClass,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2LimitationCode,
    FrvA2ProviderBindingEvidence,
    FrvA2ProviderReadiness,
    FrvA2ProviderRole,
    FrvA2ReadinessRecord,
    FrvA2ReviewCode,
    frv_a2_provider_binding_digest,
    validate_frv_a2_provider_binding_evidence,
    validate_frv_a2_readiness_record,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _component(key: str, kind: FrvA2ArtifactComponentKind, sha: str, size: int) -> dict:
    return {
        "component_key": key,
        "component_kind": kind.value,
        "sha256": sha,
        "size_bytes": size,
    }


def _legacy_components() -> list[dict]:
    return [
        _component("manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON, SHA_A, 10),
        _component("payload_npz", FrvA2ArtifactComponentKind.PAYLOAD_NPZ, SHA_B, 20),
    ]


def _i1_components(
    *, include_v2a_manifest: bool = False, include_v2a_payload: bool = False
) -> list[dict]:
    components = [
        _component(
            "bundle_manifest_json", FrvA2ArtifactComponentKind.BUNDLE_MANIFEST_JSON, SHA_A, 1
        ),
        _component(
            "v2_manifest_json", FrvA2ArtifactComponentKind.REFERENCED_MANIFEST_JSON, SHA_C, 3
        ),
    ]
    if include_v2a_manifest:
        components.append(
            _component(
                "v2a_manifest_json", FrvA2ArtifactComponentKind.REFERENCED_MANIFEST_JSON, SHA_D, 6
            )
        )
    components.extend(
        [
            _component(
                "w1_manifest_json", FrvA2ArtifactComponentKind.REFERENCED_MANIFEST_JSON, SHA_B, 2
            ),
            _component(
                "v2_payload_npz", FrvA2ArtifactComponentKind.REFERENCED_PAYLOAD_NPZ, SHA_A, 5
            ),
        ]
    )
    if include_v2a_payload:
        components.append(
            _component(
                "v2a_payload_npz", FrvA2ArtifactComponentKind.REFERENCED_PAYLOAD_NPZ, SHA_D, 7
            )
        )
    components.append(
        _component("w1_payload_npz", FrvA2ArtifactComponentKind.REFERENCED_PAYLOAD_NPZ, SHA_D, 4)
    )
    return components


def _reference_id(role: FrvA2ProviderRole, key: str, components: list[dict]) -> str:
    sequence = ";".join(
        f"{item['component_kind']},{item['component_key']},{item['sha256']},{item['size_bytes']}"
        for item in components
    )
    return f"frv-a2-ref-v1|{role.value}|{key}|{sequence}"


def _binding_payload(data: dict) -> dict:
    return {
        "digest_domain": "FRVA2BND1",
        "provider_key": data["provider_key"],
        "provider_role": data["provider_role"],
        "evidence_class": data["evidence_class"],
        "public_module": data["public_module"],
        "public_type_name": data["public_type_name"],
        "public_loader_name": data["public_loader_name"],
        "interface_version": data["interface_version"],
        "schema_version": data["schema_version"],
        "algorithm_version": data["algorithm_version"],
        "configuration_digest": data["configuration_digest"],
        "provider_config_digest": data["provider_config_digest"],
        "artifact_reference_id": data["artifact_reference_id"],
        "artifact_components": data["artifact_components"],
        "source_content_identity_digest": data["source_content_identity_digest"],
        "source_identity_record_digest": data["source_identity_record_digest"],
        "coordinate_frame": data["coordinate_frame"],
        "units": data["units"],
        "origin_xyz": data["origin_xyz"],
        "transform_declaration": data["transform_declaration"],
        "domain_kind": data["domain_kind"],
        "domain_count": data["domain_count"],
        "original_index_domain_digest": data["original_index_domain_digest"],
        "sample_index_domain_digest": data["sample_index_domain_digest"],
        "provider_readiness": data["provider_readiness"],
        "quality_flags": data["quality_flags"],
        "reason_flags": data["reason_flags"],
        "error_codes": data["error_codes"],
        "upstream_binding_digests": data["upstream_binding_digests"],
        "review_codes": data["review_codes"],
        "limitation_codes": data["limitation_codes"],
    }


def _binding(
    *,
    provider_key: str = "source_scan",
    provider_role: FrvA2ProviderRole = FrvA2ProviderRole.SOURCE_SCAN,
    evidence_class: FrvA2EvidenceClass = FrvA2EvidenceClass.FORMAL_PROVIDER,
    provider_readiness: FrvA2ProviderReadiness = FrvA2ProviderReadiness.VALIDATED,
    components: list[dict] | None = None,
    review_codes: list[str] | None = None,
    limitation_codes: list[str] | None = None,
    public_module: str = "synthetic.providers",
    public_loader_name: str = "load_synthetic_provider",
    configuration_digest: str = SHA_C,
    provider_config_digest: str | None = SHA_D,
) -> FrvA2ProviderBindingEvidence:
    selected_components = components if components is not None else _legacy_components()
    data = {
        "provider_key": provider_key,
        "provider_role": provider_role.value,
        "evidence_class": evidence_class.value,
        "public_module": public_module,
        "public_type_name": "SyntheticProvider",
        "public_loader_name": public_loader_name,
        "interface_version": "synthetic-interface-v1",
        "schema_version": None,
        "algorithm_version": "synthetic-algorithm-v1",
        "configuration_digest": configuration_digest,
        "provider_config_digest": provider_config_digest,
        "artifact_reference_id": _reference_id(provider_role, provider_key, selected_components),
        "artifact_components": selected_components,
        "source_content_identity_digest": SHA_A,
        "source_identity_record_digest": SHA_B,
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_xyz": [0.0, 0.0, 0.0],
        "transform_declaration": "identity",
        "domain_kind": FrvA2DomainKind.FULL_SOURCE.value,
        "domain_count": 2,
        "original_index_domain_digest": None,
        "sample_index_domain_digest": None,
        "provider_readiness": provider_readiness.value,
        "quality_flags": 0,
        "reason_flags": 0,
        "error_codes": [],
        "upstream_binding_digests": [],
        "review_codes": review_codes or [],
        "limitation_codes": limitation_codes or [],
    }
    data["binding_digest"] = frv_digest(_binding_payload(data))
    return FrvA2ProviderBindingEvidence.from_dict(data)


def _complete_base_providers(
    *,
    configuration_digest: str = SHA_C,
    provider_config_digest: str | None = SHA_D,
    provider_config_digests: tuple[str | None, str | None, str | None] | None = None,
) -> list[FrvA2ProviderBindingEvidence]:
    selected_provider_config_digests = provider_config_digests or (
        provider_config_digest,
        provider_config_digest,
        provider_config_digest,
    )
    return [
        _binding(
            provider_key="source_provider",
            provider_role=FrvA2ProviderRole.SOURCE_SCAN,
            configuration_digest=configuration_digest,
            provider_config_digest=selected_provider_config_digests[0],
        ),
        _binding(
            provider_key="hag_provider",
            provider_role=FrvA2ProviderRole.HAG,
            configuration_digest=configuration_digest,
            provider_config_digest=selected_provider_config_digests[1],
        ),
        _binding(
            provider_key="geometry_provider",
            provider_role=FrvA2ProviderRole.GEOMETRY,
            configuration_digest=configuration_digest,
            provider_config_digest=selected_provider_config_digests[2],
        ),
    ]


def _failure(
    *,
    phase: FrvA2FailurePhase = FrvA2FailurePhase.STRUCTURE_INTEGRITY,
    role: FrvA2ProviderRole = FrvA2ProviderRole.SOURCE_SCAN,
    key: str = "source_scan",
    field: str = "manifest",
    reason_flags: int = 0,
) -> FrvA2FailureRecord:
    error = FrvErrorCode.FRV_PROVIDER_INVALID
    detail = f"{phase.value}:{role.value}:{key}:{field}:{error.value}:{reason_flags}"
    return FrvA2FailureRecord(phase, role, key, field, error, reason_flags, detail)


def _readiness(
    providers: list[FrvA2ProviderBindingEvidence],
    aggregate: FrvA2AggregateReadiness = FrvA2AggregateReadiness.READY_FOR_A3,
    failures: list[FrvA2FailureRecord] | None = None,
    review_codes: list[str] | None = None,
    limitation_codes: list[str] | None = None,
    configuration_digest: str = SHA_C,
) -> FrvA2ReadinessRecord:
    failures = failures or []
    review_codes = review_codes or []
    limitation_codes = limitation_codes or []
    ordered = sorted(
        providers,
        key=lambda item: (list(FrvA2ProviderRole).index(item.provider_role), item.provider_key),
    )
    payload = {
        "digest_domain": "FRVA2RDY1",
        "schema_version": "frv-a2-readiness-v1",
        "source_content_identity_digest": SHA_A,
        "source_identity_record_digest": SHA_B,
        "configuration_digest": configuration_digest,
        "provider_evidence_projections": [_binding_payload(item.to_dict()) for item in ordered],
        "aggregate_readiness": aggregate.value,
        "blocking_failures": [item.to_dict() for item in failures],
        "review_codes": review_codes,
        "limitation_codes": limitation_codes,
    }
    data = {
        "schema_version": "frv-a2-readiness-v1",
        "source_content_identity_digest": SHA_A,
        "source_identity_record_digest": SHA_B,
        "configuration_digest": configuration_digest,
        "provider_evidence": [item.to_dict() for item in ordered],
        "aggregate_readiness": aggregate.value,
        "blocking_failures": [item.to_dict() for item in failures],
        "review_codes": review_codes,
        "limitation_codes": limitation_codes,
        "result_digest": frv_digest(payload),
    }
    return FrvA2ReadinessRecord.from_dict(data)


def test_a2_enum_bases_values_and_order() -> None:
    expected = {
        FrvA2ProviderRole: ["source_scan", "hag", "geometry", "wire", "provenance"],
        FrvA2EvidenceClass: ["formal_provider", "supporting_evidence"],
        FrvA2ProviderReadiness: ["validated", "review_required", "blocked", "supporting_only"],
        FrvA2AggregateReadiness: ["ready_for_a3", "ready_for_a3_with_limitations", "blocked"],
        FrvA2DomainKind: [
            "full_source",
            "declared_subset",
            "sampled_subset",
            "candidate_subset",
            "cluster_subset",
            "pair_subset",
            "artifact_reference",
            "influence_domain",
        ],
        FrvA2FailurePhase: [
            "structure_integrity",
            "identity_context",
            "domain_alignment",
            "semantic_readiness",
            "cross_provider",
            "production_policy",
        ],
        FrvA2ArtifactComponentKind: [
            "bundle_manifest_json",
            "manifest_json",
            "payload_npz",
            "source_file",
            "referenced_manifest_json",
            "referenced_payload_npz",
        ],
        FrvA2ReviewCode: ["formal_provider_review_required", "calibration_wire_config_unbound"],
        FrvA2LimitationCode: [
            "nonproduction_configuration",
            "formal_completeness_forbidden",
            "membership_publication_forbidden",
            "optional_test_wire_provider_absent",
            "formal_provider_review_required",
            "calibration_wire_config_unbound",
        ],
    }
    for enum_type, values in expected.items():
        assert issubclass(enum_type, str)
        assert issubclass(enum_type, __import__("enum").Enum)
        assert [item.value for item in enum_type] == values


def test_a2_records_have_exact_frozen_fields_and_are_frozen() -> None:
    assert [item.name for item in fields(FrvA2FailureRecord)] == [
        "phase",
        "provider_role",
        "provider_key",
        "field_name",
        "error_code",
        "reason_flags",
        "detail",
    ]
    assert [item.name for item in fields(FrvA2ArtifactComponentBinding)] == [
        "component_key",
        "component_kind",
        "sha256",
        "size_bytes",
    ]
    assert [item.name for item in fields(FrvA2ProviderBindingEvidence)] == [
        "provider_key",
        "provider_role",
        "evidence_class",
        "public_module",
        "public_type_name",
        "public_loader_name",
        "interface_version",
        "schema_version",
        "algorithm_version",
        "configuration_digest",
        "provider_config_digest",
        "artifact_reference_id",
        "artifact_components",
        "source_content_identity_digest",
        "source_identity_record_digest",
        "coordinate_frame",
        "units",
        "origin_xyz",
        "transform_declaration",
        "domain_kind",
        "domain_count",
        "original_index_domain_digest",
        "sample_index_domain_digest",
        "provider_readiness",
        "quality_flags",
        "reason_flags",
        "error_codes",
        "upstream_binding_digests",
        "review_codes",
        "limitation_codes",
        "binding_digest",
    ]
    assert [item.name for item in fields(FrvA2ReadinessRecord)] == [
        "schema_version",
        "source_content_identity_digest",
        "source_identity_record_digest",
        "configuration_digest",
        "provider_evidence",
        "aggregate_readiness",
        "blocking_failures",
        "review_codes",
        "limitation_codes",
        "result_digest",
    ]
    with pytest.raises(FrozenInstanceError):
        _binding().provider_key = "changed"  # type: ignore[misc]


def test_component_and_provider_layout_contracts() -> None:
    component = FrvA2ArtifactComponentBinding.from_dict(_legacy_components()[0])
    assert component.to_dict() == _legacy_components()[0]
    with pytest.raises(ValueError):
        FrvA2ArtifactComponentBinding("Bad-Key", FrvA2ArtifactComponentKind.MANIFEST_JSON, SHA_A, 1)
    with pytest.raises(ValueError):
        FrvA2ArtifactComponentBinding(
            "manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON, SHA_A.upper(), 1
        )
    with pytest.raises(ValueError):
        FrvA2ArtifactComponentBinding(
            "manifest_json", FrvA2ArtifactComponentKind.MANIFEST_JSON, SHA_A, True
        )
    i1_components = _i1_components()
    i1 = _binding(
        provider_key="i1",
        provider_role=FrvA2ProviderRole.PROVENANCE,
        evidence_class=FrvA2EvidenceClass.SUPPORTING_EVIDENCE,
        provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY,
        components=i1_components,
    )
    assert i1.artifact_reference_id == _reference_id(i1.provider_role, "i1", i1_components)
    validate_frv_a2_provider_binding_evidence(i1)
    i1_with_v2a = _binding(
        provider_key="i1",
        provider_role=FrvA2ProviderRole.PROVENANCE,
        evidence_class=FrvA2EvidenceClass.SUPPORTING_EVIDENCE,
        provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY,
        components=_i1_components(include_v2a_manifest=True, include_v2a_payload=True),
    )
    validate_frv_a2_provider_binding_evidence(i1_with_v2a)
    for manifest, payload in ((True, False), (False, True)):
        partial = i1_with_v2a
        partial_components = _i1_components(
            include_v2a_manifest=manifest,
            include_v2a_payload=payload,
        )
        object.__setattr__(
            partial,
            "artifact_components",
            tuple(FrvA2ArtifactComponentBinding.from_dict(item) for item in partial_components),
        )
        object.__setattr__(partial, "binding_digest", frv_a2_provider_binding_digest(partial))
        with pytest.raises(ValueError):
            validate_frv_a2_provider_binding_evidence(partial)
    with pytest.raises(ValueError):
        _binding(provider_key="a2b", provider_role=FrvA2ProviderRole.GEOMETRY,
                 evidence_class=FrvA2EvidenceClass.SUPPORTING_EVIDENCE,
                 provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY)


def test_supporting_mapping_and_formal_gate() -> None:
    evidence = _binding(
        provider_key="a2b",
        provider_role=FrvA2ProviderRole.HAG,
        evidence_class=FrvA2EvidenceClass.SUPPORTING_EVIDENCE,
        provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY,
    )
    assert evidence.provider_readiness is FrvA2ProviderReadiness.SUPPORTING_ONLY
    with pytest.raises(ValueError):
        _binding(
            evidence_class=FrvA2EvidenceClass.FORMAL_PROVIDER,
            provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY,
        )


def test_failure_detail_sorting_and_deduplication() -> None:
    first = _failure(reason_flags=1)
    second = _failure(reason_flags=2)
    assert first.detail.endswith(":1")
    assert second.detail.endswith(":2")
    record = _readiness(
        [_binding()],
        aggregate=FrvA2AggregateReadiness.BLOCKED,
        failures=[first, second],
    )
    assert record.blocking_failures == (first, second)
    with pytest.raises(ValueError):
        _readiness(
            [_binding()],
            aggregate=FrvA2AggregateReadiness.BLOCKED,
            failures=[first, first],
        )
    with pytest.raises(ValueError):
        FrvA2FailureRecord(
            first.phase,
            first.provider_role,
            first.provider_key,
            first.field_name,
            first.error_code,
            first.reason_flags,
            "free text",
        )


def test_readiness_aggregate_invariants_and_no_runtime_fields() -> None:
    ready = _readiness(_complete_base_providers())
    validate_frv_a2_readiness_record(ready)
    limited = _readiness(
        _complete_base_providers(),
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        limitation_codes=[FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION.value],
    )
    assert limited.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS
    with pytest.raises(ValueError):
        _readiness([], aggregate=FrvA2AggregateReadiness.BLOCKED)
    with pytest.raises(ValueError):
        _readiness(_complete_base_providers(), aggregate=FrvA2AggregateReadiness.READY_FOR_A3,
                   review_codes=[FrvA2ReviewCode.FORMAL_PROVIDER_REVIEW_REQUIRED.value])
    assert "path" not in ready.to_dict()
    assert "runtime" not in ready.to_dict()


def test_base_formal_role_gate_is_required_without_configuration() -> None:
    with pytest.raises(ValueError):
        _readiness([_binding()])
    with pytest.raises(ValueError):
        _readiness(
            [
                _binding(),
                _binding(
                    provider_key="geometry_provider", provider_role=FrvA2ProviderRole.GEOMETRY
                ),
            ]
        )
    with pytest.raises(ValueError):
        _readiness(
            [
                _binding(),
                _binding(provider_key="hag_provider", provider_role=FrvA2ProviderRole.HAG),
            ]
        )
    with pytest.raises(ValueError):
        _readiness(
            [
                _binding(),
                _binding(
                    provider_key="a2b",
                    provider_role=FrvA2ProviderRole.HAG,
                    evidence_class=FrvA2EvidenceClass.SUPPORTING_EVIDENCE,
                    provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY,
                ),
                _binding(
                    provider_key="geometry_provider", provider_role=FrvA2ProviderRole.GEOMETRY
                ),
            ]
        )
    with pytest.raises(ValueError):
        _binding(
            provider_key="hag_provider",
            provider_role=FrvA2ProviderRole.HAG,
            provider_readiness=FrvA2ProviderReadiness.SUPPORTING_ONLY,
        )
    with pytest.raises(ValueError):
        _readiness(
            _complete_base_providers()
            + [_binding(provider_key="hag_duplicate", provider_role=FrvA2ProviderRole.HAG)]
        )
    assert (
        _readiness(_complete_base_providers()).aggregate_readiness
        is FrvA2AggregateReadiness.READY_FOR_A3
    )


def test_numpy_and_bool_strictness() -> None:
    with pytest.raises(ValueError):
        FrvA2FailureRecord(
            FrvA2FailurePhase.STRUCTURE_INTEGRITY,
            FrvA2ProviderRole.SOURCE_SCAN,
            "source_scan",
            "field",
            FrvErrorCode.FRV_PROVIDER_INVALID,
            np.int64(0),
            "structure_integrity:source_scan:source_scan:field:FRV_PROVIDER_INVALID:0",
        )
    with pytest.raises(FrozenInstanceError):
        _binding().artifact_components = []  # type: ignore[misc]


def test_test_only_half_bound_rule_is_public_and_exact_in_both_directions() -> None:
    from tests.contracts.test_frv_contracts import config_dict

    for wire_interface, wire_digest in (("wire-v1", None), (None, SHA_D)):
        config = FrvConfiguration.from_dict(
            config_dict(
                configuration_purpose="TEST_ONLY",
                configuration_is_runtime_complete=False,
                formal_completeness_allowed=False,
                membership_artifact_publication_allowed=False,
                wire_provider_interface_version=wire_interface,
                wire_provider_config_digest=wire_digest,
            )
        )
        failure = _failure(
            phase=FrvA2FailurePhase.PRODUCTION_POLICY,
            role=FrvA2ProviderRole.WIRE,
            key="wire",
            field="wire_provider_binding",
            reason_flags=int(FrvReasonFlag.PROVIDER_UNCERTAINTY),
        )
        blocked = _readiness(
            [],
            aggregate=FrvA2AggregateReadiness.BLOCKED,
            failures=[failure],
            configuration_digest=config.configuration_digest,
        )
        validate_frv_a2_readiness_record(blocked, config)
        assert blocked.blocking_failures == (failure,)
        assert failure.error_code is FrvErrorCode.FRV_PROVIDER_INVALID
        assert failure.detail == (
            "production_policy:wire:wire:wire_provider_binding:FRV_PROVIDER_INVALID:"
            f"{int(FrvReasonFlag.PROVIDER_UNCERTAINTY)}"
        )
        limited = _readiness(
            _complete_base_providers(configuration_digest=config.configuration_digest),
            aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
            limitation_codes=[FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION.value],
            configuration_digest=config.configuration_digest,
        )
        with pytest.raises(ValueError):
            validate_frv_a2_readiness_record(limited, config)


def test_test_only_optional_wire_absence_policy() -> None:
    from tests.contracts.test_frv_contracts import config_dict

    absent_config = FrvConfiguration.from_dict(
        config_dict(
            configuration_purpose="TEST_ONLY",
            configuration_is_runtime_complete=False,
            formal_completeness_allowed=False,
            membership_artifact_publication_allowed=False,
            wire_provider_interface_version=None,
            wire_provider_config_digest=None,
        )
    )
    nonproduction_limitations = [
        FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION.value,
        FrvA2LimitationCode.FORMAL_COMPLETENESS_FORBIDDEN.value,
        FrvA2LimitationCode.MEMBERSHIP_PUBLICATION_FORBIDDEN.value,
    ]
    missing_optional = _readiness(
        _complete_base_providers(
            configuration_digest=absent_config.configuration_digest,
            provider_config_digests=(
                absent_config.scan_provider_config_digest,
                absent_config.hag_provider_config_digest,
                absent_config.geometry_provider_config_digest,
            ),
        ),
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        limitation_codes=nonproduction_limitations,
        configuration_digest=absent_config.configuration_digest,
    )
    with pytest.raises(ValueError):
        validate_frv_a2_readiness_record(missing_optional, absent_config)
    valid_absent = _readiness(
        _complete_base_providers(
            configuration_digest=absent_config.configuration_digest,
            provider_config_digests=(
                absent_config.scan_provider_config_digest,
                absent_config.hag_provider_config_digest,
                absent_config.geometry_provider_config_digest,
            ),
        ),
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        limitation_codes=nonproduction_limitations
        + [FrvA2LimitationCode.OPTIONAL_TEST_WIRE_PROVIDER_ABSENT.value],
        configuration_digest=absent_config.configuration_digest,
    )
    validate_frv_a2_readiness_record(valid_absent, absent_config)
    bound_config = FrvConfiguration.from_dict(
        config_dict(
            configuration_purpose="TEST_ONLY",
            configuration_is_runtime_complete=False,
            formal_completeness_allowed=False,
            membership_artifact_publication_allowed=False,
            wire_provider_interface_version="wire-v1",
            wire_provider_config_digest=SHA_D,
        )
    )
    bound_wire = _binding(
        provider_key="wire_provider",
        provider_role=FrvA2ProviderRole.WIRE,
        configuration_digest=bound_config.configuration_digest,
        provider_config_digest=bound_config.wire_provider_config_digest,
    )
    bound = _readiness(
        _complete_base_providers(
            configuration_digest=bound_config.configuration_digest,
            provider_config_digests=(
                bound_config.scan_provider_config_digest,
                bound_config.hag_provider_config_digest,
                bound_config.geometry_provider_config_digest,
            ),
        )
        + [bound_wire],
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        limitation_codes=nonproduction_limitations,
        configuration_digest=bound_config.configuration_digest,
    )
    validate_frv_a2_readiness_record(bound, bound_config)
    with_optional = _readiness(
        _complete_base_providers(
            configuration_digest=bound_config.configuration_digest,
            provider_config_digests=(
                bound_config.scan_provider_config_digest,
                bound_config.hag_provider_config_digest,
                bound_config.geometry_provider_config_digest,
            ),
        )
        + [bound_wire],
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        limitation_codes=nonproduction_limitations
        + [FrvA2LimitationCode.OPTIONAL_TEST_WIRE_PROVIDER_ABSENT.value],
        configuration_digest=bound_config.configuration_digest,
    )
    with pytest.raises(ValueError):
        validate_frv_a2_readiness_record(with_optional, bound_config)


def test_test_only_full_wire_binding_rejects_missing_formal_wire_provider() -> None:
    from tests.contracts.test_frv_contracts import config_dict

    config = FrvConfiguration.from_dict(
        config_dict(
            configuration_purpose="TEST_ONLY",
            configuration_is_runtime_complete=False,
            formal_completeness_allowed=False,
            membership_artifact_publication_allowed=False,
            wire_provider_interface_version="wire-v1",
            wire_provider_config_digest=SHA_D,
        )
    )
    nonproduction_limitations = [
        FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION.value,
        FrvA2LimitationCode.FORMAL_COMPLETENESS_FORBIDDEN.value,
        FrvA2LimitationCode.MEMBERSHIP_PUBLICATION_FORBIDDEN.value,
    ]
    assert FrvA2LimitationCode.OPTIONAL_TEST_WIRE_PROVIDER_ABSENT.value not in (
        nonproduction_limitations
    )
    record = _readiness(
        _complete_base_providers(
            configuration_digest=config.configuration_digest,
            provider_config_digests=(
                config.scan_provider_config_digest,
                config.hag_provider_config_digest,
                config.geometry_provider_config_digest,
            ),
        ),
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        limitation_codes=nonproduction_limitations,
        configuration_digest=config.configuration_digest,
    )
    with pytest.raises(ValueError, match="required formal role wire is not uniquely bound"):
        validate_frv_a2_readiness_record(record, config)


def test_test_only_null_wire_binding_rejects_ready_for_a3() -> None:
    from tests.contracts.test_frv_contracts import config_dict

    config = FrvConfiguration.from_dict(
        config_dict(
            configuration_purpose="TEST_ONLY",
            configuration_is_runtime_complete=False,
            formal_completeness_allowed=False,
            membership_artifact_publication_allowed=False,
            wire_provider_interface_version=None,
            wire_provider_config_digest=None,
        )
    )
    record = _readiness(
        _complete_base_providers(
            configuration_digest=config.configuration_digest,
            provider_config_digests=(
                config.scan_provider_config_digest,
                config.hag_provider_config_digest,
                config.geometry_provider_config_digest,
            ),
        ),
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3,
        limitation_codes=[],
        configuration_digest=config.configuration_digest,
    )
    with pytest.raises(ValueError, match="non-production configuration limitations are incomplete"):
        validate_frv_a2_readiness_record(record, config)


def test_configuration_aware_required_roles_and_calibration_policy() -> None:
    from tests.contracts.test_frv_contracts import config_dict

    production = FrvConfiguration.from_dict(config_dict())
    production_providers = [
        _binding(
            provider_key="source_provider",
            provider_role=FrvA2ProviderRole.SOURCE_SCAN,
            configuration_digest=production.configuration_digest,
            provider_config_digest=production.scan_provider_config_digest,
        ),
        _binding(
            provider_key="hag_provider",
            provider_role=FrvA2ProviderRole.HAG,
            configuration_digest=production.configuration_digest,
            provider_config_digest=production.hag_provider_config_digest,
        ),
        _binding(
            provider_key="geometry_provider",
            provider_role=FrvA2ProviderRole.GEOMETRY,
            configuration_digest=production.configuration_digest,
            provider_config_digest=production.geometry_provider_config_digest,
        ),
        _binding(
            provider_key="wire_provider",
            provider_role=FrvA2ProviderRole.WIRE,
            configuration_digest=production.configuration_digest,
            provider_config_digest=production.wire_provider_config_digest,
        ),
    ]
    production_record = _readiness(
        production_providers, configuration_digest=production.configuration_digest
    )
    validate_frv_a2_readiness_record(production_record, production)

    calibration = FrvConfiguration.from_dict(
        config_dict(
            configuration_purpose="CALIBRATION_CANDIDATE",
            configuration_is_runtime_complete=False,
            formal_completeness_allowed=False,
            membership_artifact_publication_allowed=False,
            wire_provider_interface_version="wire-v1",
            wire_provider_config_digest=None,
        )
    )
    calibration_base_providers = [
        _binding(
            provider_key="source_provider",
            provider_role=FrvA2ProviderRole.SOURCE_SCAN,
            configuration_digest=calibration.configuration_digest,
            provider_config_digest=calibration.scan_provider_config_digest,
        ),
        _binding(
            provider_key="hag_provider",
            provider_role=FrvA2ProviderRole.HAG,
            configuration_digest=calibration.configuration_digest,
            provider_config_digest=calibration.hag_provider_config_digest,
        ),
        _binding(
            provider_key="geometry_provider",
            provider_role=FrvA2ProviderRole.GEOMETRY,
            configuration_digest=calibration.configuration_digest,
            provider_config_digest=calibration.geometry_provider_config_digest,
        ),
    ]
    calibration_providers = calibration_base_providers + [
        _binding(
            provider_key="wire_provider",
            provider_role=FrvA2ProviderRole.WIRE,
            provider_readiness=FrvA2ProviderReadiness.REVIEW_REQUIRED,
            provider_config_digest=None,
            configuration_digest=calibration.configuration_digest,
        )
    ]
    calibration_limitation_codes = [
        FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION.value,
        FrvA2LimitationCode.FORMAL_COMPLETENESS_FORBIDDEN.value,
        FrvA2LimitationCode.MEMBERSHIP_PUBLICATION_FORBIDDEN.value,
        FrvA2LimitationCode.CALIBRATION_WIRE_CONFIG_UNBOUND.value,
    ]
    calibration_record = _readiness(
        calibration_providers,
        aggregate=FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS,
        review_codes=[FrvA2ReviewCode.CALIBRATION_WIRE_CONFIG_UNBOUND.value],
        limitation_codes=calibration_limitation_codes,
        configuration_digest=calibration.configuration_digest,
    )
    validate_frv_a2_readiness_record(calibration_record, calibration)
