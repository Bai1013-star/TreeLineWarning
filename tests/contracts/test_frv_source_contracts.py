"""Focused FRV-A2 P-SOURCE C1 contract tests.

C1 covers only the frozen public constants, closed enums, and the exact-shape
immutable contracts.  No laspy, no file access, no streaming digest, no
failure mapping, and no A2 BND1 adapter is exercised here.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_source as source
import numpy as np
import pytest
from core.contracts.frv import canonical_frv_json_bytes
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_source import (
    FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN,
    FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
    FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_KEY,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourceCanonicalContentPolicy,
    FrvSourceChunk,
    FrvSourceCompressionState,
    FrvSourceCoordinateConversionPolicy,
    FrvSourceCrsAuthorityState,
    FrvSourceFormat,
    FrvSourceHeaderMetadata,
    FrvSourceInvalidPointPolicy,
    FrvSourceModel,
    FrvSourcePassThroughDimension,
    FrvSourceProviderConfig,
    FrvSourceProviderReference,
    FrvSourceVlrPolicyState,
)

SHA = "a" * 64

_FIXTURE_SOURCE_PATH_TEXT = "fixtures/source.las"
_FIXTURE_SOURCE_PATH = Path(_FIXTURE_SOURCE_PATH_TEXT)
_ARTIFACT_REFERENCE_ID = (
    "frv-a2-ref-v1|source_scan|source_scan|source_file,source_file," + SHA + ",123"
)


def _source_identity() -> frv.SourceIdentity:
    relocation_payload = {
        "audit_schema_version": frv.FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows",
        "recorded_path_raw": _FIXTURE_SOURCE_PATH_TEXT,
        "resolved_path_raw": _FIXTURE_SOURCE_PATH_TEXT,
        "recorded_path_lexical_normalized": _FIXTURE_SOURCE_PATH_TEXT,
        "resolved_path_lexical_normalized": _FIXTURE_SOURCE_PATH_TEXT,
        "resolved_final_path": None,
        "resolution_method": "exact_match",
        "case_comparison_mode": "case_insensitive",
        "platform_name": "windows",
        "run_id": "run-1",
        "audit_timestamp": "2026-01-01T00:00:00Z",
        "content_identity_result": "matched",
        "warning_flags": 0,
    }
    relocation_payload["audit_digest"] = frv.frv_digest(
        {key: value for key, value in relocation_payload.items()}
    )
    relocation = frv.FrvRelocationAudit.from_dict(relocation_payload)
    content_payload = {
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": "scene-1",
        "source_file_sha256": SHA,
        "canonical_point_content_digest": None,
        "canonical_point_content_spec_digest": None,
        "canonical_point_content_spec_version": None,
        "source_point_count": 3,
        "source_bounds_min_xyz": [0.0, 0.0, 0.0],
        "source_bounds_max_xyz": [1.0, 1.0, 1.0],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_context": [0.0, 0.0, 0.0],
    }
    content_digest = frv.frv_digest(content_payload)
    record_payload = {
        **content_payload,
        "recorded_source_path": _FIXTURE_SOURCE_PATH_TEXT,
        "resolved_access_path": _FIXTURE_SOURCE_PATH_TEXT,
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return frv.SourceIdentity.from_dict(
        {
            **content_payload,
            "recorded_source_path": _FIXTURE_SOURCE_PATH_TEXT,
            "resolved_access_path": _FIXTURE_SOURCE_PATH_TEXT,
            "relocation_audit": relocation.to_dict(),
            "source_equivalence_audit": None,
            "source_content_identity_digest": content_digest,
            "source_identity_record_digest": frv.frv_digest(record_payload),
        }
    )


def _canonical_spec() -> frv.FrvCanonicalPointContentSpec:
    payload = {
        "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "spec_id": "point-spec-1",
        "framing_version": "framing-v1",
        "row_order": "ORIGINAL_INDEX_ASCENDING",
        "ordered_field_specs": [
            {
                "field_name": "original_index",
                "semantic_role": "ORIGINAL_INDEX",
                "dtype_code": "<i8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            }
        ],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "quantization_policy": "EXACT_TYPED_VALUE",
        "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
    }
    payload["spec_digest"] = frv.frv_digest(payload)
    return frv.FrvCanonicalPointContentSpec.from_dict(payload)


def _component() -> FrvA2ArtifactComponentBinding:
    return FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=SHA,
        size_bytes=123,
    )


def _config(**overrides: Any) -> FrvSourceProviderConfig:
    defaults: dict[str, Any] = {
        "schema_version": source.FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        "source_model": FrvSourceModel.SINGLE_FILE_ONLY,
        "coordinate_conversion_policy": FrvSourceCoordinateConversionPolicy.FORBID,
        "invalid_point_policy": FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        "canonical_content_policy": FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        "pass_through_dimensions": (),
        "provider_config_digest": SHA,
    }
    defaults.update(overrides)
    return FrvSourceProviderConfig(**defaults)


def _header(**overrides: Any) -> FrvSourceHeaderMetadata:
    defaults: dict[str, Any] = {
        "schema_version": source.FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION,
        "source_format": FrvSourceFormat.LAS,
        "las_version": "1.2",
        "point_format_id": 3,
        "point_record_length_bytes": 34,
        "point_count": 3,
        "scale_xyz": (0.01, 0.01, 0.01),
        "offset_xyz": (1000.0, 2000.0, 50.0),
        "min_xyz": (0.0, 0.0, 0.0),
        "max_xyz": (1.0, 1.0, 1.0),
        "dimensions": ("classification", "intensity"),
        "extra_dimensions": (),
        "compression_state": FrvSourceCompressionState.UNCOMPRESSED,
        "crs_authority_state": FrvSourceCrsAuthorityState.ABSENT,
        "non_crs_vlr_count": 0,
        "non_crs_evlr_count": 0,
        "vlr_policy_state": FrvSourceVlrPolicyState.NO_CRS_AUTHORITY_DETECTED,
    }
    defaults.update(overrides)
    return FrvSourceHeaderMetadata(**defaults)


def _reference(**overrides: Any) -> FrvSourceProviderReference:
    defaults: dict[str, Any] = {
        "provider_role": FrvA2ProviderRole.SOURCE_SCAN,
        "provider_key": source.FRV_SOURCE_PROVIDER_KEY,
        "artifact_reference_id": _ARTIFACT_REFERENCE_ID,
        "artifact_components": (_component(),),
        "runtime_access_path": _FIXTURE_SOURCE_PATH,
        "source_identity": _source_identity(),
        "canonical_point_content_spec": None,
        "configuration_digest": SHA,
        "provider_config_digest": "b" * 64,
        "interface_version": source.FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        "schema_version": source.FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        "algorithm_version": source.FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    }
    defaults.update(overrides)
    return FrvSourceProviderReference(**defaults)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _chunk(**overrides: Any) -> FrvSourceChunk:
    defaults: dict[str, Any] = {
        "source_id": "scene-1",
        "chunk_id": 0,
        "start_index": 0,
        "point_count": 2,
        "xyz": _readonly(
            np.ascontiguousarray(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8"))
        ),
        "original_index": _readonly(
            np.ascontiguousarray(np.array([0, 1], dtype="<i8"))
        ),
        "attributes": MappingProxyType(
            {
                "classification": _readonly(
                    np.ascontiguousarray(np.array([1, 2], dtype="|u1"))
                ),
                "intensity": _readonly(
                    np.ascontiguousarray(np.array([100, 200], dtype="<u2"))
                ),
            }
        ),
        "is_last": False,
    }
    defaults.update(overrides)
    return FrvSourceChunk(**defaults)


# ---------------------------------------------------------------------------
# 1. Public constants
# ---------------------------------------------------------------------------


def test_public_constants_exact_names_and_values() -> None:
    assert source.FRV_SOURCE_PROVIDER_KEY == "source_scan"
    assert source.FRV_SOURCE_PROVIDER_INTERFACE_VERSION == "frv-source-provider-v1"
    assert source.FRV_SOURCE_PROVIDER_SCHEMA_VERSION == "frv-source-provider-result-v1"
    assert source.FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION == "frv-source-provider-config-v1"
    assert source.FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION == "frv-source-header-v1"
    assert source.FRV_SOURCE_PROVIDER_ALGORITHM_VERSION == "frv-source-laspy-single-handle-v1"
    assert source.FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN == "FRVSRCFG1"
    assert source.FRV_SOURCE_DEFAULT_CHUNK_SIZE == 1000000


# ---------------------------------------------------------------------------
# 2. Closed enums: members, values, declaration order
# ---------------------------------------------------------------------------


def test_single_member_enums_exact_values() -> None:
    assert FrvSourceModel.SINGLE_FILE_ONLY == "single_file_only"
    assert FrvSourceCoordinateConversionPolicy.FORBID == "forbid"
    assert FrvSourceInvalidPointPolicy.FAIL_CLOSED == "fail_closed"
    assert FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED == "verify_if_declared"
    assert FrvSourceFormat.LAS == "las"
    assert FrvSourceCompressionState.UNCOMPRESSED == "uncompressed"
    assert FrvSourceCrsAuthorityState.ABSENT == "absent"
    assert FrvSourceVlrPolicyState.NO_CRS_AUTHORITY_DETECTED == "no_crs_authority_detected"


def test_pass_through_dimension_declaration_order() -> None:
    assert [member.value for member in FrvSourcePassThroughDimension] == [
        "classification",
        "intensity",
        "return_number",
        "number_of_returns",
        "scan_angle",
        "gps_time",
        "point_source_id",
        "red",
        "green",
        "blue",
    ]
    assert list(FrvSourcePassThroughDimension) == [
        FrvSourcePassThroughDimension.CLASSIFICATION,
        FrvSourcePassThroughDimension.INTENSITY,
        FrvSourcePassThroughDimension.RETURN_NUMBER,
        FrvSourcePassThroughDimension.NUMBER_OF_RETURNS,
        FrvSourcePassThroughDimension.SCAN_ANGLE,
        FrvSourcePassThroughDimension.GPS_TIME,
        FrvSourcePassThroughDimension.POINT_SOURCE_ID,
        FrvSourcePassThroughDimension.RED,
        FrvSourcePassThroughDimension.GREEN,
        FrvSourcePassThroughDimension.BLUE,
    ]


def test_enums_reject_unknown_value_and_wrong_case() -> None:
    with pytest.raises(ValueError):
        FrvSourceModel("multi_file")
    with pytest.raises(ValueError):
        FrvSourceFormat("LAZ")
    with pytest.raises(ValueError):
        FrvSourcePassThroughDimension("unknown")
    with pytest.raises(ValueError):
        FrvSourcePassThroughDimension("CLASSIFICATION")
    with pytest.raises(ValueError):
        FrvSourceVlrPolicyState("cr_authority_detected")


# ---------------------------------------------------------------------------
# 3. Frozen, slots, exact fields
# ---------------------------------------------------------------------------


def test_contracts_are_frozen_slots_and_exact_fields() -> None:
    config = _config()
    assert hasattr(config, "__slots__")
    with pytest.raises(FrozenInstanceError):
        config.schema_version = "x"  # type: ignore[misc]
    assert set(FrvSourceProviderConfig.__dataclass_fields__) == {
        "schema_version",
        "source_model",
        "coordinate_conversion_policy",
        "invalid_point_policy",
        "canonical_content_policy",
        "pass_through_dimensions",
        "provider_config_digest",
    }

    header = _header()
    assert hasattr(header, "__slots__")
    with pytest.raises(FrozenInstanceError):
        header.point_count = 5  # type: ignore[misc]
    assert set(FrvSourceHeaderMetadata.__dataclass_fields__) == {
        "schema_version",
        "source_format",
        "las_version",
        "point_format_id",
        "point_record_length_bytes",
        "point_count",
        "scale_xyz",
        "offset_xyz",
        "min_xyz",
        "max_xyz",
        "dimensions",
        "extra_dimensions",
        "compression_state",
        "crs_authority_state",
        "non_crs_vlr_count",
        "non_crs_evlr_count",
        "vlr_policy_state",
    }

    reference = _reference()
    assert hasattr(reference, "__slots__")
    with pytest.raises(FrozenInstanceError):
        reference.provider_key = "x"  # type: ignore[misc]
    assert set(FrvSourceProviderReference.__dataclass_fields__) == {
        "provider_role",
        "provider_key",
        "artifact_reference_id",
        "artifact_components",
        "runtime_access_path",
        "source_identity",
        "canonical_point_content_spec",
        "configuration_digest",
        "provider_config_digest",
        "interface_version",
        "schema_version",
        "algorithm_version",
    }

    chunk = _chunk()
    assert hasattr(chunk, "__slots__")
    with pytest.raises(FrozenInstanceError):
        chunk.chunk_id = 1  # type: ignore[misc]
    assert set(FrvSourceChunk.__dataclass_fields__) == {
        "source_id",
        "chunk_id",
        "start_index",
        "point_count",
        "xyz",
        "original_index",
        "attributes",
        "is_last",
    }


# ---------------------------------------------------------------------------
# 4. FrvSourceProviderConfig
# ---------------------------------------------------------------------------


def test_config_normal_construction_and_round_trip() -> None:
    config = _config(
        pass_through_dimensions=(
            FrvSourcePassThroughDimension.CLASSIFICATION,
            FrvSourcePassThroughDimension.INTENSITY,
        )
    )
    assert config.schema_version == source.FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION
    assert config.provider_config_digest == SHA
    restored = FrvSourceProviderConfig.from_dict(config.to_dict())
    assert restored == config
    assert restored.pass_through_dimensions == (
        FrvSourcePassThroughDimension.CLASSIFICATION,
        FrvSourcePassThroughDimension.INTENSITY,
    )


def test_config_rejects_wrong_schema_version() -> None:
    with pytest.raises(ValueError, match="schema version"):
        _config(schema_version="frv-source-provider-config-v2")


def test_config_rejects_bare_string_enum() -> None:
    with pytest.raises(ValueError, match="source_model"):
        _config(source_model="single_file_only")


def test_config_rejects_list_dimensions() -> None:
    with pytest.raises(ValueError, match="immutable tuple"):
        _config(pass_through_dimensions=[FrvSourcePassThroughDimension.CLASSIFICATION])


def test_config_rejects_duplicate_dimensions() -> None:
    with pytest.raises(ValueError, match="unique"):
        _config(
            pass_through_dimensions=(
                FrvSourcePassThroughDimension.CLASSIFICATION,
                FrvSourcePassThroughDimension.CLASSIFICATION,
            )
        )


def test_config_rejects_unsorted_dimensions() -> None:
    with pytest.raises(ValueError, match="declaration order"):
        _config(
            pass_through_dimensions=(
                FrvSourcePassThroughDimension.INTENSITY,
                FrvSourcePassThroughDimension.CLASSIFICATION,
            )
        )


def test_config_rejects_unknown_dimension_via_from_dict() -> None:
    payload = _config().to_dict()
    payload["pass_through_dimensions"] = ["unknown"]
    with pytest.raises(ValueError, match="unsupported"):
        FrvSourceProviderConfig.from_dict(payload)


def test_config_rejects_bad_digest() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        _config(provider_config_digest="A" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        _config(provider_config_digest="a" * 63)


def test_config_from_dict_rejects_unknown_and_missing_fields() -> None:
    payload = _config().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown fields"):
        FrvSourceProviderConfig.from_dict(payload)
    del payload["extra"]
    del payload["schema_version"]
    with pytest.raises(ValueError, match="missing fields"):
        FrvSourceProviderConfig.from_dict(payload)


# ---------------------------------------------------------------------------
# 5. FrvSourceHeaderMetadata
# ---------------------------------------------------------------------------


def test_header_normal_construction_and_round_trip() -> None:
    header = _header()
    assert header.schema_version == source.FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION
    assert header.scale_xyz == (0.01, 0.01, 0.01)
    restored = FrvSourceHeaderMetadata.from_dict(header.to_dict())
    assert restored == header
    assert isinstance(restored.scale_xyz, tuple)
    assert restored.dimensions == ("classification", "intensity")


def test_header_rejects_bool_as_int() -> None:
    for name in ("point_format_id", "point_count", "non_crs_vlr_count", "non_crs_evlr_count"):
        with pytest.raises(ValueError, match=name):
            _header(**{name: True})
    with pytest.raises(ValueError, match="point_record_length_bytes"):
        _header(point_record_length_bytes=False)


def test_header_rejects_negative_counts() -> None:
    with pytest.raises(ValueError, match="point_count"):
        _header(point_count=-1)
    with pytest.raises(ValueError, match="non_crs_vlr_count"):
        _header(non_crs_vlr_count=-1)
    with pytest.raises(ValueError, match="point_format_id"):
        _header(point_format_id=-1)


def test_header_rejects_non_positive_record_length() -> None:
    with pytest.raises(ValueError, match="point_record_length_bytes"):
        _header(point_record_length_bytes=0)


def test_header_rejects_invalid_tuple_length() -> None:
    with pytest.raises(ValueError, match="exactly three"):
        _header(scale_xyz=(0.01, 0.01))


def test_header_rejects_list_instead_of_tuple() -> None:
    with pytest.raises(ValueError, match="immutable tuple"):
        _header(offset_xyz=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="immutable tuple"):
        _header(dimensions=["classification"])


def test_header_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError, match="finite"):
        _header(scale_xyz=(math.nan, 0.01, 0.01))
    with pytest.raises(ValueError, match="finite"):
        _header(max_xyz=(math.inf, 1.0, 1.0))


def test_header_rejects_unsorted_duplicate_uppercase_dimensions() -> None:
    with pytest.raises(ValueError, match="lexicographically"):
        _header(dimensions=("intensity", "classification"))
    with pytest.raises(ValueError, match="unique"):
        _header(dimensions=("classification", "classification"))
    with pytest.raises(ValueError, match="lower-case"):
        _header(dimensions=("Classification",))


def test_header_rejects_overlapping_standard_and_extra_dimensions() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        _header(extra_dimensions=("intensity",))


def test_header_rejects_min_exceeding_max() -> None:
    with pytest.raises(ValueError, match="must not exceed max_xyz"):
        _header(min_xyz=(2.0, 0.0, 0.0), max_xyz=(1.0, 1.0, 1.0))


def test_header_allows_zero_point_count() -> None:
    header = _header(point_count=0)
    assert header.point_count == 0


# ---------------------------------------------------------------------------
# 6. FrvSourceProviderReference (local shape/type only)
# ---------------------------------------------------------------------------


def test_reference_normal_local_construction() -> None:
    reference = _reference()
    assert reference.provider_role is FrvA2ProviderRole.SOURCE_SCAN
    assert reference.artifact_reference_id == _ARTIFACT_REFERENCE_ID
    assert reference.runtime_access_path == _FIXTURE_SOURCE_PATH
    assert reference.source_identity.source_id == "scene-1"
    assert reference.canonical_point_content_spec is None


def test_reference_accepts_a_real_canonical_spec() -> None:
    reference = _reference(canonical_point_content_spec=_canonical_spec())
    assert reference.canonical_point_content_spec is not None
    assert reference.canonical_point_content_spec.spec_id == "point-spec-1"


def test_reference_artifact_reference_id_accepts_a2_grammar() -> None:
    reference = _reference()
    assert reference.artifact_reference_id.startswith("frv-a2-ref-v1|")
    assert reference.artifact_reference_id == (
        "frv-a2-ref-v1|source_scan|source_scan|source_file,source_file," + SHA + ",123"
    )


def test_reference_artifact_reference_id_rejects_bad_values() -> None:
    with pytest.raises(TypeError, match="built-in str"):
        _reference(artifact_reference_id=b"frv-a2-ref-v1|source_scan")
    with pytest.raises(ValueError, match="non-empty"):
        _reference(artifact_reference_id="")
    with pytest.raises(ValueError, match="whitespace"):
        _reference(artifact_reference_id="   ")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        _reference(artifact_reference_id=" frv-a2-ref-v1|source_scan")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        _reference(artifact_reference_id="frv-a2-ref-v1|source_scan ")
    with pytest.raises(ValueError, match="must start with"):
        _reference(artifact_reference_id="some-other-prefix|source_scan")


def test_reference_rejects_wrong_field_types() -> None:
    with pytest.raises(ValueError, match="provider_role"):
        _reference(provider_role="source_scan")
    with pytest.raises(TypeError, match="provider_key"):
        _reference(provider_key=123)
    with pytest.raises(ValueError, match="runtime_access_path"):
        _reference(runtime_access_path=_FIXTURE_SOURCE_PATH_TEXT)
    with pytest.raises(ValueError, match="source_identity"):
        _reference(source_identity="not-an-identity")
    with pytest.raises(ValueError, match="canonical_point_content_spec"):
        _reference(canonical_point_content_spec="not-a-spec")
    with pytest.raises(ValueError, match="interface_version"):
        _reference(interface_version="")
    with pytest.raises(TypeError, match="schema_version"):
        _reference(schema_version=5)
    with pytest.raises(ValueError, match="algorithm_version"):
        _reference(algorithm_version="")


def test_reference_rejects_bad_digests() -> None:
    with pytest.raises(ValueError, match="configuration_digest"):
        _reference(configuration_digest="not-a-sha")
    with pytest.raises(ValueError, match="provider_config_digest"):
        _reference(provider_config_digest="B" * 64)


def test_reference_rejects_non_tuple_components() -> None:
    with pytest.raises(ValueError, match="immutable tuple"):
        _reference(artifact_components=[_component()])
    with pytest.raises(ValueError, match="FrvA2ArtifactComponentBinding"):
        _reference(artifact_components=(_component(), "not-a-component"))


def test_reference_is_not_canonically_serializable() -> None:
    reference = _reference()
    assert not hasattr(reference, "to_dict")
    assert not hasattr(reference, "from_dict")
    with pytest.raises(ValueError, match="FrvSourceProviderReference"):
        canonical_frv_json_bytes(reference)


# ---------------------------------------------------------------------------
# 7. FrvSourceChunk
# ---------------------------------------------------------------------------


def test_chunk_normal_construction() -> None:
    chunk = _chunk()
    assert chunk.source_id == "scene-1"
    assert chunk.chunk_id == 0
    assert chunk.start_index == 0
    assert chunk.point_count == 2
    assert chunk.xyz.shape == (2, 3)
    assert chunk.original_index.tolist() == [0, 1]
    assert chunk.is_last is False


def test_chunk_rejects_bool_as_int() -> None:
    for name in ("chunk_id", "start_index", "point_count"):
        with pytest.raises(ValueError, match=name):
            _chunk(**{name: True})
    with pytest.raises(ValueError, match="is_last"):
        _chunk(is_last=1)


def test_chunk_rejects_negative_integers() -> None:
    with pytest.raises(ValueError, match="chunk_id"):
        _chunk(chunk_id=-1)
    with pytest.raises(ValueError, match="start_index"):
        _chunk(start_index=-1)
    with pytest.raises(ValueError, match="point_count"):
        _chunk(point_count=-1)


def test_chunk_rejects_writable_array() -> None:
    writable_xyz = np.ascontiguousarray(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8")
    )
    with pytest.raises(ValueError, match="read-only"):
        _chunk(xyz=writable_xyz)
    writable_index = np.ascontiguousarray(np.array([0, 1], dtype="<i8"))
    with pytest.raises(ValueError, match="read-only"):
        _chunk(original_index=writable_index)


def test_chunk_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="dtype <f8"):
        _chunk(xyz=_readonly(np.ascontiguousarray(np.array([[0.0, 0.0, 0.0]], dtype="<f4"))))
    with pytest.raises(ValueError, match="dtype <i8"):
        _chunk(original_index=_readonly(np.ascontiguousarray(np.array([0], dtype="<i4"))))


def test_chunk_rejects_wrong_endianness() -> None:
    big_endian_xyz = _readonly(
        np.ascontiguousarray(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=">f8"))
    )
    with pytest.raises(ValueError, match="dtype <f8"):
        _chunk(xyz=big_endian_xyz)
    big_endian_intensity = _readonly(
        np.ascontiguousarray(np.array([100, 200], dtype=">u2"))
    )
    with pytest.raises(ValueError, match="dtype <u2"):
        _chunk(
            attributes=MappingProxyType({"intensity": big_endian_intensity})
        )


def test_chunk_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        _chunk(xyz=_readonly(np.ascontiguousarray(np.array([[0.0, 0.0]], dtype="<f8"))))
    with pytest.raises(ValueError, match="shape"):
        _chunk(original_index=_readonly(np.ascontiguousarray(np.array([[0]], dtype="<i8"))))


def test_chunk_rejects_non_contiguous_array() -> None:
    non_contiguous = np.zeros((3, 2), dtype="<f8").T
    non_contiguous.setflags(write=False)
    with pytest.raises(ValueError, match="C-contiguous"):
        _chunk(xyz=non_contiguous)


def test_chunk_rejects_nonfinite_xyz() -> None:
    bad_xyz = _readonly(
        np.ascontiguousarray(
            np.array([[math.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8")
        )
    )
    with pytest.raises(ValueError, match="finite"):
        _chunk(xyz=bad_xyz)


def test_chunk_rejects_nonfinite_float_attributes() -> None:
    attributes = MappingProxyType(
        {
            "gps_time": _readonly(
                np.ascontiguousarray(np.array([math.inf, 1.0], dtype="<f8"))
            )
        }
    )
    with pytest.raises(ValueError, match="finite"):
        _chunk(attributes=attributes)
    attributes = MappingProxyType(
        {
            "scan_angle": _readonly(
                np.ascontiguousarray(np.array([math.nan, 1.0], dtype="<f8"))
            )
        }
    )
    with pytest.raises(ValueError, match="finite"):
        _chunk(attributes=attributes)


def test_chunk_rejects_mutable_attributes_mapping() -> None:
    with pytest.raises(ValueError, match="immutable mapping"):
        _chunk(
            attributes={
                "classification": _readonly(
                    np.ascontiguousarray(np.array([1], dtype="|u1"))
                )
            }
        )
    with pytest.raises(ValueError, match="immutable mapping"):
        _chunk(attributes=[("classification", np.array([1], dtype="|u1"))])


def test_chunk_rejects_unknown_attribute_key() -> None:
    attributes = MappingProxyType(
        {
            "not_a_dimension": _readonly(
                np.ascontiguousarray(np.array([1, 2], dtype="|u1"))
            )
        }
    )
    with pytest.raises(ValueError, match="unknown pass-through dimension"):
        _chunk(attributes=attributes)


def test_chunk_rejects_attribute_order_wrong() -> None:
    attributes = MappingProxyType(
        {
            "intensity": _readonly(np.ascontiguousarray(np.array([100, 200], dtype="<u2"))),
            "classification": _readonly(np.ascontiguousarray(np.array([1, 2], dtype="|u1"))),
        }
    )
    with pytest.raises(ValueError, match="declaration order"):
        _chunk(attributes=attributes)


def test_chunk_rejects_attribute_dtype_mismatch() -> None:
    attributes = MappingProxyType(
        {
            "classification": _readonly(
                np.ascontiguousarray(np.array([1, 2], dtype="<u2"))
            )
        }
    )
    with pytest.raises(ValueError, match="dtype"):
        _chunk(attributes=attributes)


def test_chunk_rejects_attribute_point_count_mismatch() -> None:
    attributes = MappingProxyType(
        {
            "classification": _readonly(
                np.ascontiguousarray(np.array([1, 2, 3], dtype="|u1"))
            )
        }
    )
    with pytest.raises(ValueError, match="shape"):
        _chunk(attributes=attributes)


def test_chunk_attributes_are_immutable_externally() -> None:
    chunk = _chunk()
    with pytest.raises(TypeError):
        chunk.attributes["classification"] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        chunk.attributes["new_key"] = None  # type: ignore[index]


def test_chunk_is_not_canonically_serializable() -> None:
    chunk = _chunk()
    assert not hasattr(chunk, "to_dict")
    assert not hasattr(chunk, "from_dict")
    with pytest.raises(ValueError, match="FrvSourceChunk"):
        canonical_frv_json_bytes(chunk)


# ---------------------------------------------------------------------------
# 8. Public exports and absence of unauthorized future symbols
# ---------------------------------------------------------------------------


def test_public_symbols_importable_from_core_contracts() -> None:
    imported = (
        FRV_SOURCE_PROVIDER_KEY,
        FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION,
        FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
        FRV_SOURCE_PROVIDER_CONFIG_DIGEST_DOMAIN,
        FRV_SOURCE_DEFAULT_CHUNK_SIZE,
        FrvSourceModel,
        FrvSourceCoordinateConversionPolicy,
        FrvSourceInvalidPointPolicy,
        FrvSourceCanonicalContentPolicy,
        FrvSourceFormat,
        FrvSourceCompressionState,
        FrvSourceCrsAuthorityState,
        FrvSourceVlrPolicyState,
        FrvSourcePassThroughDimension,
        FrvSourceProviderConfig,
        FrvSourceHeaderMetadata,
        FrvSourceProviderReference,
        FrvSourceChunk,
    )
    assert all(item is not None for item in imported)
    assert contracts.FrvSourceProviderConfig is source.FrvSourceProviderConfig
    assert contracts.FrvSourcePassThroughDimension is source.FrvSourcePassThroughDimension
    assert contracts.FRV_SOURCE_PROVIDER_KEY == "source_scan"


def test_no_unauthorized_future_public_symbols() -> None:
    for name in (
        "FrvSourceLoadSession",
        "FrvSourceProviderProof",
        "FrvSourceProviderResult",
        "FrvSourceLoadError",
        "load_frv_source",
        "validate_frv_source_provider_reference",
        "validate_finalized_frv_source_provider_result",
        "to_frv_a2_provider_binding_evidence",
        "FrvSourceProviderProof",
    ):
        assert not hasattr(contracts, name)


# ---------------------------------------------------------------------------
# 9. Independent-review fixes (F-001 alias safety, F-002 str subclass)
# ---------------------------------------------------------------------------


class SchemaVersionSubclass(str):
    """Minimal test-only str subclass; never used in production code."""


def test_chunk_xyz_has_no_external_mutable_alias() -> None:
    original = np.ascontiguousarray(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8")
    )
    original.setflags(write=False)
    chunk = _chunk(xyz=original)
    # Re-enabling the caller's array must not reach the stored chunk.
    original.setflags(write=True)
    original[0, 0] = 999.0
    assert chunk.xyz[0, 0] == 0.0
    # The stored array is bytes-backed and cannot be re-enabled for writing.
    with pytest.raises(ValueError):
        chunk.xyz.setflags(write=True)


def test_chunk_original_index_has_no_external_mutable_alias() -> None:
    original = np.ascontiguousarray(np.array([0, 1], dtype="<i8"))
    original.setflags(write=False)
    chunk = _chunk(original_index=original)
    original.setflags(write=True)
    original[0] = 42
    assert chunk.original_index.tolist() == [0, 1]
    with pytest.raises(ValueError):
        chunk.original_index.setflags(write=True)


def test_chunk_attributes_have_no_external_mutable_alias() -> None:
    backing = {
        "classification": _readonly(
            np.ascontiguousarray(np.array([1, 2], dtype="|u1"))
        )
    }
    attributes = MappingProxyType(backing)
    chunk = _chunk(attributes=attributes)
    # Mutating the caller's backing dict must not reach the stored chunk.
    backing["classification"] = _readonly(
        np.ascontiguousarray(np.array([9, 9], dtype="|u1"))
    )
    assert chunk.attributes["classification"].tolist() == [1, 2]
    # The stored attribute arrays are bytes-backed and cannot be re-enabled.
    with pytest.raises(ValueError):
        chunk.attributes["classification"].setflags(write=True)
    # The stored mapping itself is immutable.
    with pytest.raises(TypeError):
        chunk.attributes["new_key"] = None  # type: ignore[index]


def test_config_rejects_str_subclass_schema_version() -> None:
    with pytest.raises(TypeError, match="built-in str"):
        _config(
            schema_version=SchemaVersionSubclass(
                source.FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION
            )
        )


def test_header_rejects_str_subclass_schema_version() -> None:
    with pytest.raises(TypeError, match="built-in str"):
        _header(
            schema_version=SchemaVersionSubclass(
                source.FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION
            )
        )


class _AlwaysEqualSchema:
    """Test-only object whose __eq__ always returns True."""

    def __eq__(self, other: object) -> bool:
        return True


@pytest.mark.parametrize(
    ("factory", "bad_value"),
    [
        (_config, b"frv-source-provider-config-v1"),
        (_header, b"frv-source-header-v1"),
    ],
)
def test_non_builtin_str_schema_version_raises_type_error(
    factory: Any, bad_value: Any
) -> None:
    for value in (bad_value, None, _AlwaysEqualSchema()):
        with pytest.raises(TypeError, match="built-in str"):
            factory(schema_version=value)


@pytest.mark.parametrize(
    ("factory", "wrong_schema"),
    [
        (_config, "frv-source-provider-config-v2"),
        (_header, "frv-source-header-v2"),
    ],
)
def test_wrong_schema_value_still_raises_value_error(
    factory: Any, wrong_schema: str
) -> None:
    with pytest.raises(ValueError, match="schema version"):
        factory(schema_version=wrong_schema)


# -- F-003: writable-base view alias safety regression -------------------------


def test_chunk_xyz_writable_base_view_no_alias() -> None:
    base = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype="<f8")
    view = base.view()
    view.setflags(write=False)
    chunk = _chunk(xyz=view)
    base[0, 0] = 999.0
    assert chunk.xyz[0, 0] == 0.0
    assert np.shares_memory(chunk.xyz, base) is False
    assert np.shares_memory(chunk.xyz, view) is False
    assert chunk.xyz.flags.writeable is False
    assert chunk.xyz.shape == (2, 3)
    assert chunk.xyz.dtype == np.dtype("<f8")
    assert chunk.xyz.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        chunk.xyz.setflags(write=True)


def test_chunk_original_index_writable_base_view_no_alias() -> None:
    base = np.array([0, 1], dtype="<i8")
    view = base.view()
    view.setflags(write=False)
    chunk = _chunk(original_index=view)
    base[0] = 42
    assert chunk.original_index.tolist() == [0, 1]
    assert np.shares_memory(chunk.original_index, base) is False
    assert np.shares_memory(chunk.original_index, view) is False
    assert chunk.original_index.flags.writeable is False
    assert chunk.original_index.dtype == np.dtype("<i8")
    with pytest.raises(ValueError):
        chunk.original_index.setflags(write=True)


def test_chunk_attribute_writable_base_view_no_alias() -> None:
    base = np.array([1, 2], dtype="|u1")
    view = base.view()
    view.setflags(write=False)
    attributes = MappingProxyType({"classification": view})
    chunk = _chunk(attributes=attributes)
    base[:] = [9, 9]
    assert chunk.attributes["classification"].tolist() == [1, 2]
    assert np.shares_memory(chunk.attributes["classification"], base) is False
    assert np.shares_memory(chunk.attributes["classification"], view) is False
    assert chunk.attributes["classification"].flags.writeable is False
    assert chunk.attributes["classification"].dtype == np.dtype("|u1")
    assert chunk.attributes["classification"].flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        chunk.attributes["classification"].setflags(write=True)


# -- F-003: mapping backing-dict isolation regression --------------------------


def test_chunk_attributes_backing_dict_isolated() -> None:
    classification_base = np.array([1, 2], dtype="|u1")
    classification_view = classification_base.view()
    classification_view.setflags(write=False)
    intensity_base = np.array([100, 200], dtype="<u2")
    intensity_view = intensity_base.view()
    intensity_view.setflags(write=False)
    backing = {"classification": classification_view}
    chunk = _chunk(attributes=MappingProxyType(backing))

    # Replace an existing key in the caller's backing dict.
    replacement = _readonly(np.ascontiguousarray(np.array([7, 7], dtype="|u1")))
    backing["classification"] = replacement
    assert chunk.attributes["classification"] is not replacement
    assert chunk.attributes["classification"].tolist() == [1, 2]

    # Add a new key in the caller's backing dict.
    backing["intensity"] = intensity_view
    assert "intensity" not in chunk.attributes

    # Delete an existing key from the caller's backing dict.
    del backing["classification"]
    assert "classification" in chunk.attributes
    assert chunk.attributes["classification"].tolist() == [1, 2]

    # Mutate the writable base behind the original attribute view.
    classification_base[:] = [5, 5]
    assert chunk.attributes["classification"].tolist() == [1, 2]
    assert np.shares_memory(chunk.attributes["classification"], classification_base) is False
    assert np.shares_memory(chunk.attributes["classification"], classification_view) is False

    # Direct mutation of the stored mapping must fail.
    with pytest.raises(TypeError):
        chunk.attributes["classification"] = replacement  # type: ignore[index]
    with pytest.raises(TypeError):
        chunk.attributes["new_key"] = None  # type: ignore[index]
    with pytest.raises(ValueError):
        chunk.attributes["classification"].setflags(write=True)
