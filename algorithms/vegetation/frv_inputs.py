"""Live finalized FRV-A2 provider bundle (PLAN-FROZEN).

``validate_frv_a2_inputs`` validates the four finalized formal providers and
their four ``FRVA2BND1`` adapters, derives the frozen A2 readiness record, and
returns an immutable no-copy capability bundle.  It never reopens a provider or
source, never selects a path, never constructs a scope, and never decides
membership.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.frv import (
    FrvConfiguration,
    FrvConfigurationPurpose,
    validate_configuration,
)
from core.contracts.frv_a2 import (
    FrvA2AggregateReadiness,
    FrvA2LimitationCode,
    FrvA2ProviderBindingEvidence,
    FrvA2ProviderReadiness,
    FrvA2ProviderRole,
    FrvA2ReadinessRecord,
    frv_a2_readiness_record_digest,
    validate_frv_a2_readiness_record,
)
from core.contracts.frv_geometry import (
    FrvGeometryProviderResult,
    to_frv_a2_geometry_provider_binding_evidence,
    validate_finalized_frv_geometry_provider_result,
)
from core.contracts.frv_hag import (
    FrvHagProviderResult,
    to_frv_a2_hag_provider_binding_evidence,
    validate_finalized_frv_hag_provider_result,
)
from core.contracts.frv_source import (
    FrvSourceProviderResult,
    to_frv_a2_provider_binding_evidence,
    validate_finalized_frv_source_provider_result,
)
from core.contracts.frv_wire import (
    FrvWireProviderResult,
    to_frv_a2_wire_provider_binding_evidence,
    validate_finalized_frv_wire_provider_result,
)

_READINESS_SCHEMA_VERSION = "frv-a2-readiness-v1"

_NON_PRODUCTION_LIMITATIONS = (
    FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION,
    FrvA2LimitationCode.FORMAL_COMPLETENESS_FORBIDDEN,
    FrvA2LimitationCode.MEMBERSHIP_PUBLICATION_FORBIDDEN,
)


def _role_order(role: FrvA2ProviderRole) -> int:
    return {
        FrvA2ProviderRole.SOURCE_SCAN: 0,
        FrvA2ProviderRole.HAG: 1,
        FrvA2ProviderRole.GEOMETRY: 2,
        FrvA2ProviderRole.WIRE: 3,
    }[role]


def _sort_key(evidence: FrvA2ProviderBindingEvidence) -> tuple[int, str]:
    return (_role_order(evidence.provider_role), evidence.provider_key)


@dataclass(frozen=True, slots=True)
class ValidatedFrvA2Inputs:
    """Immutable no-copy bundle of the four live finalized providers."""

    source_result: FrvSourceProviderResult
    hag_result: FrvHagProviderResult
    geometry_result: FrvGeometryProviderResult
    wire_result: FrvWireProviderResult
    readiness_record: FrvA2ReadinessRecord


def validate_frv_a2_inputs(
    source_result: FrvSourceProviderResult,
    hag_result: FrvHagProviderResult,
    geometry_result: FrvGeometryProviderResult,
    wire_result: FrvWireProviderResult,
    *,
    frv_configuration: FrvConfiguration,
) -> ValidatedFrvA2Inputs:
    """Validate the four finalized providers and derive the frozen A2 readiness.

    Any non-validated provider, identity mismatch, or failed readiness gate
    raises ``ValueError``; a partial bundle is never returned.
    """

    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(hag_result) is not FrvHagProviderResult:
        raise TypeError("hag_result must be a FrvHagProviderResult")
    if type(geometry_result) is not FrvGeometryProviderResult:
        raise TypeError("geometry_result must be a FrvGeometryProviderResult")
    if type(wire_result) is not FrvWireProviderResult:
        raise TypeError("wire_result must be a FrvWireProviderResult")
    if type(frv_configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")

    validate_finalized_frv_source_provider_result(source_result)
    validate_finalized_frv_hag_provider_result(hag_result)
    validate_finalized_frv_geometry_provider_result(geometry_result)
    validate_finalized_frv_wire_provider_result(wire_result)
    validate_configuration(frv_configuration)

    evidence = tuple(
        sorted(
            (
                to_frv_a2_provider_binding_evidence(source_result),
                to_frv_a2_hag_provider_binding_evidence(hag_result),
                to_frv_a2_geometry_provider_binding_evidence(geometry_result),
                to_frv_a2_wire_provider_binding_evidence(wire_result),
            ),
            key=_sort_key,
        )
    )
    source = source_result.source_identity
    for item in evidence:
        if item.source_content_identity_digest != source.source_content_identity_digest:
            raise ValueError("provider evidence is not bound to the same source content")
        if item.source_identity_record_digest != source.source_identity_record_digest:
            raise ValueError("provider evidence is not bound to the same source record")
        if item.provider_readiness is not FrvA2ProviderReadiness.VALIDATED:
            raise ValueError(f"{item.provider_key} is not validated; no partial bundle")

    purpose = frv_configuration.configuration_purpose
    limitation_codes: tuple[FrvA2LimitationCode, ...] = ()
    if purpose is FrvConfigurationPurpose.CALIBRATION_CANDIDATE or purpose is (
        FrvConfigurationPurpose.TEST_ONLY
    ):
        limitation_codes = _NON_PRODUCTION_LIMITATIONS

    aggregate = (
        FrvA2AggregateReadiness.READY_FOR_A3
        if not limitation_codes
        else FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS
    )

    record = FrvA2ReadinessRecord.__new__(FrvA2ReadinessRecord)
    object.__setattr__(record, "schema_version", _READINESS_SCHEMA_VERSION)
    object.__setattr__(
        record,
        "source_content_identity_digest",
        source.source_content_identity_digest,
    )
    object.__setattr__(
        record,
        "source_identity_record_digest",
        source.source_identity_record_digest,
    )
    object.__setattr__(
        record, "configuration_digest", frv_configuration.configuration_digest
    )
    object.__setattr__(record, "provider_evidence", evidence)
    object.__setattr__(record, "aggregate_readiness", aggregate)
    object.__setattr__(record, "blocking_failures", ())
    object.__setattr__(record, "review_codes", ())
    object.__setattr__(record, "limitation_codes", limitation_codes)
    object.__setattr__(record, "result_digest", frv_a2_readiness_record_digest(record))
    validate_frv_a2_readiness_record(record, frv_configuration)

    if (
        purpose is FrvConfigurationPurpose.PRODUCTION_APPROVED
        and aggregate is not FrvA2AggregateReadiness.READY_FOR_A3
    ):
        raise ValueError("production configuration requires READY_FOR_A3 readiness")

    return ValidatedFrvA2Inputs(
        source_result=source_result,
        hag_result=hag_result,
        geometry_result=geometry_result,
        wire_result=wire_result,
        readiness_record=record,
    )
