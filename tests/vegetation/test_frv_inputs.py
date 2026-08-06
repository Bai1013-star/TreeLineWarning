"""Live FRV-A2 provider bundle tests (PLAN-FROZEN)."""

from __future__ import annotations

from pathlib import Path

import core.contracts.frv as frv
import pytest
from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation import (
    load_frv_geometry,
    load_frv_hag,
    load_frv_wire,
)
from algorithms.vegetation.frv_inputs import validate_frv_a2_inputs
from core.contracts.frv_a2 import (
    FrvA2AggregateReadiness,
    FrvA2LimitationCode,
    FrvA2ProviderRole,
    validate_frv_a2_readiness_record,
)
from core.contracts.frv_source import FrvSourcePassThroughDimension

from tests.contracts.test_frv_geometry_contracts import _config as _geometry_config
from tests.contracts.test_frv_hag_contracts import _config as _hag_config
from tests.contracts.test_frv_wire_contracts import _config as _wire_config
from tests.io.test_source_provider import _configuration, _provider_config
from tests.vegetation.test_hag_provider import write_hag_las
from tests.vegetation.test_wire_provider import (
    _build_authority,
    _source_reference,
    _w1_arrays,
)


def _configuration_all(source_config, hag_config, geometry_config, wire_config):
    base = _configuration().to_digest_dict()
    base["scan_provider_config_digest"] = source_config.provider_config_digest
    base["hag_provider_interface_version"] = hag_config.interface_version
    base["hag_provider_config_digest"] = hag_config.provider_config_digest
    base["geometry_provider_interface_version"] = geometry_config.interface_version
    base["geometry_provider_config_digest"] = geometry_config.provider_config_digest
    base["wire_provider_interface_version"] = wire_config.interface_version
    base["wire_provider_config_digest"] = wire_config.provider_config_digest
    base["configuration_digest"] = frv.frv_digest(base)
    return frv.FrvConfiguration.from_dict(base)


def _run_pipeline(tmp_path: Path, *, chunk_size: int = 3):
    """Run source -> hag -> geometry -> wire over one LAS and return the results."""
    las_path = tmp_path / "s.las"
    write_hag_las(las_path, point_count=8)
    geometry_config = _geometry_config()
    wire_config = _wire_config()
    hag_config = _hag_config()
    source_config = _provider_config(
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,)
    )
    configuration = _configuration_all(
        source_config, hag_config, geometry_config, wire_config
    )
    reference = _source_reference(las_path, source_config, configuration)
    with load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=source_config,
        chunk_size=chunk_size,
    ) as session:
        list(session.iter_chunks())
        source_result = session.finalize()
    with load_frv_hag(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=hag_config,
        source_chunk_size=chunk_size,
    ) as session:
        list(session.iter_chunks())
        hag_result = session.finalize()
    with load_frv_geometry(
        reference,
        source_result,
        frv_configuration=configuration,
        provider_config=geometry_config,
        scratch_dir=tmp_path,
        source_chunk_size=chunk_size,
    ) as session:
        list(session.iter_chunks())
        geometry_result = session.finalize()
    arrays = _w1_arrays()
    authority, _ = _build_authority(tmp_path, source_result, arrays=arrays)
    with load_frv_wire(
        reference,
        source_result,
        authority,
        frv_configuration=configuration,
        provider_config=wire_config,
        source_chunk_size=chunk_size,
    ) as session:
        list(session.iter_chunks())
        wire_result = session.finalize()
    return (
        configuration,
        source_result,
        hag_result,
        geometry_result,
        wire_result,
    )


def test_bundle_validates_and_holds_references(tmp_path: Path) -> None:
    configuration, source_result, hag_result, geometry_result, wire_result = (
        _run_pipeline(tmp_path)
    )
    bundle = validate_frv_a2_inputs(
        source_result,
        hag_result,
        geometry_result,
        wire_result,
        frv_configuration=configuration,
    )
    assert bundle.source_result is source_result
    assert bundle.hag_result is hag_result
    assert bundle.geometry_result is geometry_result
    assert bundle.wire_result is wire_result
    record = bundle.readiness_record
    validate_frv_a2_readiness_record(record, configuration)
    roles = {item.provider_role for item in record.provider_evidence}
    assert roles == {
        FrvA2ProviderRole.SOURCE_SCAN,
        FrvA2ProviderRole.HAG,
        FrvA2ProviderRole.GEOMETRY,
        FrvA2ProviderRole.WIRE,
    }
    assert record.aggregate_readiness is FrvA2AggregateReadiness.READY_FOR_A3_WITH_LIMITATIONS
    assert (
        FrvA2LimitationCode.NONPRODUCTION_CONFIGURATION in record.limitation_codes
    )
    assert record.blocking_failures == ()


def test_bundle_is_readiness_complete(tmp_path: Path) -> None:
    configuration, source_result, hag_result, geometry_result, wire_result = (
        _run_pipeline(tmp_path)
    )
    bundle = validate_frv_a2_inputs(
        source_result,
        hag_result,
        geometry_result,
        wire_result,
        frv_configuration=configuration,
    )
    record = bundle.readiness_record
    assert record.result_digest == frv.frv_digest(
        {
            "digest_domain": "FRVA2RDY1",
            "schema_version": record.schema_version,
            "source_content_identity_digest": record.source_content_identity_digest,
            "source_identity_record_digest": record.source_identity_record_digest,
            "configuration_digest": record.configuration_digest,
            "provider_evidence_projections": [
                {
                    "digest_domain": "FRVA2BND1",
                    "provider_key": item.provider_key,
                    "provider_role": item.provider_role.value,
                    "evidence_class": item.evidence_class.value,
                    "public_module": item.public_module,
                    "public_type_name": item.public_type_name,
                    "public_loader_name": item.public_loader_name,
                    "interface_version": item.interface_version,
                    "schema_version": item.schema_version,
                    "algorithm_version": item.algorithm_version,
                    "configuration_digest": item.configuration_digest,
                    "provider_config_digest": item.provider_config_digest,
                    "artifact_reference_id": item.artifact_reference_id,
                    "artifact_components": [
                        component.to_dict()
                        for component in item.artifact_components
                    ],
                    "source_content_identity_digest": item.source_content_identity_digest,
                    "source_identity_record_digest": item.source_identity_record_digest,
                    "coordinate_frame": item.coordinate_frame,
                    "units": item.units,
                    "origin_xyz": list(item.origin_xyz),
                    "transform_declaration": item.transform_declaration,
                    "domain_kind": item.domain_kind.value,
                    "domain_count": item.domain_count,
                    "original_index_domain_digest": item.original_index_domain_digest,
                    "sample_index_domain_digest": item.sample_index_domain_digest,
                    "provider_readiness": item.provider_readiness.value,
                    "quality_flags": item.quality_flags,
                    "reason_flags": item.reason_flags,
                    "error_codes": [code.value for code in item.error_codes],
                    "upstream_binding_digests": list(item.upstream_binding_digests),
                    "review_codes": [code.value for code in item.review_codes],
                    "limitation_codes": [code.value for code in item.limitation_codes],
                }
                for item in record.provider_evidence
            ],
            "aggregate_readiness": record.aggregate_readiness.value,
            "blocking_failures": [
                failure.to_dict() for failure in record.blocking_failures
            ],
            "review_codes": [code.value for code in record.review_codes],
            "limitation_codes": [code.value for code in record.limitation_codes],
        }
    )


def test_bundle_rejects_identity_mismatch(tmp_path: Path) -> None:
    configuration, source_result, hag_result, geometry_result, wire_result = (
        _run_pipeline(tmp_path)
    )
    mismatched = source_result.__class__.__new__(source_result.__class__)
    object.__setattr__(mismatched, "proof", source_result.proof)
    object.__setattr__(mismatched, "source_identity", hag_result.source_identity)
    with pytest.raises(ValueError):
        validate_frv_a2_inputs(
            mismatched,
            hag_result,
            geometry_result,
            wire_result,
            frv_configuration=configuration,
        )


def test_bundle_rejects_wrong_types(tmp_path: Path) -> None:
    configuration, source_result, hag_result, geometry_result, wire_result = (
        _run_pipeline(tmp_path)
    )
    with pytest.raises(TypeError):
        validate_frv_a2_inputs(
            source_result,
            hag_result,
            geometry_result,
            None,  # type: ignore[arg-type]
            frv_configuration=configuration,
        )
