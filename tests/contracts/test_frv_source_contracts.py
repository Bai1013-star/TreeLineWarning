"""Focused FRV-A2 P-SOURCE C1/C2 contract tests.

C1 covers the frozen public constants, closed enums, and exact-shape immutable
contracts.  C2 covers the FRVSRCFG1 provider-config digest, the FRVPCD1 and
FRVA2IDX1 streaming digests, and the full P-SOURCE reference validator.  No
laspy, no file access, no failure mapping, and no A2 BND1 adapter is exercised.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import struct
import weakref
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest import mock

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_source as source
import numpy as np
import pytest
from core.contracts.frv import (
    FrvErrorCode,
    FrvQualityFlag,
    canonical_frv_json_bytes,
    validate_frv_flag_mask,
)
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
    frv_a2_index_domain_digest,
    validate_frv_a2_failure_record,
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
    FrvSourceLoadError,
    FrvSourceLoadSession,
    FrvSourceModel,
    FrvSourcePassThroughDimension,
    FrvSourceProviderConfig,
    FrvSourceProviderReference,
    FrvSourceVlrPolicyState,
    derive_frv_source_original_index_digest,
    derive_frv_source_point_content_digest,
    derive_frv_source_provider_config_digest,
    validate_frv_source_provider_reference,
)

SHA = "a" * 64

_FIXTURE_SOURCE_PATH_TEXT = "fixtures/source.las"
_FIXTURE_SOURCE_PATH = Path(_FIXTURE_SOURCE_PATH_TEXT)

# Synthetic absolute Windows paths used ONLY by the reference-validator and
# path-comparison tests.  These are pure lexical fixture data: they are never
# accessed, created, or resolved, and no drive letter is required to exist.
_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT = r"C:\__frv_source_contract_test__\source.las"
_SYNTHETIC_WINDOWS_SOURCE_PATH = Path(_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT)
_SYNTHETIC_WINDOWS_OTHER_PATH_TEXT = r"C:\__frv_source_contract_test__\other.las"
_SYNTHETIC_WINDOWS_OTHER_PATH = Path(_SYNTHETIC_WINDOWS_OTHER_PATH_TEXT)

_ARTIFACT_REFERENCE_ID = (
    "frv-a2-ref-v1|source_scan|source_scan|source_file,source_file," + SHA + ",123"
)


def _source_identity(
    *,
    path: str = _FIXTURE_SOURCE_PATH_TEXT,
    canonical_digest: str | None = None,
    canonical_spec_digest: str | None = None,
    canonical_spec_version: str | None = None,
) -> frv.SourceIdentity:
    relocation_payload = {
        "audit_schema_version": frv.FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows",
        "recorded_path_raw": path,
        "resolved_path_raw": path,
        "recorded_path_lexical_normalized": path,
        "resolved_path_lexical_normalized": path,
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
        "canonical_point_content_digest": canonical_digest,
        "canonical_point_content_spec_digest": canonical_spec_digest,
        "canonical_point_content_spec_version": canonical_spec_version,
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
        "recorded_source_path": path,
        "resolved_access_path": path,
        "relocation_audit_audit_digest": relocation.audit_digest,
        "source_equivalence_audit_audit_digest": None,
        "source_content_identity_digest": content_digest,
    }
    return frv.SourceIdentity.from_dict(
        {
            **content_payload,
            "recorded_source_path": path,
            "resolved_access_path": path,
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
    }
    explicit_digest = overrides.pop("provider_config_digest", None)
    defaults.update(overrides)
    if explicit_digest is not None:
        defaults["provider_config_digest"] = explicit_digest
    else:
        defaults["provider_config_digest"] = source.derive_frv_source_provider_config_digest(
            schema_version=defaults["schema_version"],
            source_model=defaults["source_model"],
            coordinate_conversion_policy=defaults["coordinate_conversion_policy"],
            invalid_point_policy=defaults["invalid_point_policy"],
            canonical_content_policy=defaults["canonical_content_policy"],
            pass_through_dimensions=defaults["pass_through_dimensions"],
        )
    return FrvSourceProviderConfig(**defaults)


def _exact_canonical_spec(spec_id: str = "point-spec-1") -> frv.FrvCanonicalPointContentSpec:
    payload = {
        "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "spec_id": spec_id,
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
            },
            {
                "field_name": "x",
                "semantic_role": "X_COORDINATE",
                "dtype_code": "<f8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            },
            {
                "field_name": "y",
                "semantic_role": "Y_COORDINATE",
                "dtype_code": "<f8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            },
            {
                "field_name": "z",
                "semantic_role": "Z_COORDINATE",
                "dtype_code": "<f8",
                "byte_order": "LITTLE_ENDIAN",
                "scale": 1.0,
                "offset": 0.0,
                "required": True,
                "null_policy": "FORBID_NULL",
            },
        ],
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "quantization_policy": "EXACT_TYPED_VALUE",
        "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
    }
    payload["spec_digest"] = frv.frv_digest(payload)
    return frv.FrvCanonicalPointContentSpec.from_dict(payload)


def _chunk_with_points(
    xyz_values: list[list[float]],
    original_index_values: list[int],
    *,
    source_id: str = "scene-1",
    chunk_id: int = 0,
    start_index: int = 0,
    is_last: bool = True,
) -> FrvSourceChunk:
    xyz = _readonly(np.ascontiguousarray(np.array(xyz_values, dtype="<f8")))
    original = _readonly(
        np.ascontiguousarray(np.array(original_index_values, dtype="<i8"))
    )
    return FrvSourceChunk(
        source_id=source_id,
        chunk_id=chunk_id,
        start_index=start_index,
        point_count=len(original_index_values),
        xyz=xyz,
        original_index=original,
        attributes=MappingProxyType({}),
        is_last=is_last,
    )


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
    assert config.provider_config_digest == derive_frv_source_provider_config_digest(
        schema_version=config.schema_version,
        source_model=config.source_model,
        coordinate_conversion_policy=config.coordinate_conversion_policy,
        invalid_point_policy=config.invalid_point_policy,
        canonical_content_policy=config.canonical_content_policy,
        pass_through_dimensions=config.pass_through_dimensions,
    )
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
        FrvSourceLoadError,
        FrvSourceLoadSession,
    )
    assert all(item is not None for item in imported)
    assert contracts.FrvSourceProviderConfig is source.FrvSourceProviderConfig
    assert contracts.FrvSourcePassThroughDimension is source.FrvSourcePassThroughDimension
    assert contracts.FRV_SOURCE_PROVIDER_KEY == "source_scan"
    assert contracts.FrvSourceLoadError is source.FrvSourceLoadError
    assert contracts.FrvSourceLoadSession is source.FrvSourceLoadSession


def test_no_unauthorized_future_public_symbols() -> None:
    for name in (
        "FrvSourceProviderProof",
        "FrvSourceProviderResult",
        "validate_finalized_frv_source_provider_result",
        "to_frv_a2_provider_binding_evidence",
        "load_frv_source",
        "_FRV_SOURCE_FINALIZED_CAPABILITY",
        "_create_finalized_frv_source_provider_result",
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


# ---------------------------------------------------------------------------
# 10. C2: FRVSRCFG1 provider-config digest
# ---------------------------------------------------------------------------


def _expected_spec_header_bytes(spec_id: str = "point-spec-1") -> bytes:
    """Independently build the frozen canonical point-content spec JSON bytes."""
    payload = {
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "framing_version": "framing-v1",
        "missing_value_policy": "EXPLICIT_VALIDITY_BYTE",
        "ordered_field_specs": [
            {
                "byte_order": "LITTLE_ENDIAN",
                "dtype_code": "<i8",
                "field_name": "original_index",
                "null_policy": "FORBID_NULL",
                "offset": 0.0,
                "required": True,
                "scale": 1.0,
                "semantic_role": "ORIGINAL_INDEX",
            },
            {
                "byte_order": "LITTLE_ENDIAN",
                "dtype_code": "<f8",
                "field_name": "x",
                "null_policy": "FORBID_NULL",
                "offset": 0.0,
                "required": True,
                "scale": 1.0,
                "semantic_role": "X_COORDINATE",
            },
            {
                "byte_order": "LITTLE_ENDIAN",
                "dtype_code": "<f8",
                "field_name": "y",
                "null_policy": "FORBID_NULL",
                "offset": 0.0,
                "required": True,
                "scale": 1.0,
                "semantic_role": "Y_COORDINATE",
            },
            {
                "byte_order": "LITTLE_ENDIAN",
                "dtype_code": "<f8",
                "field_name": "z",
                "null_policy": "FORBID_NULL",
                "offset": 0.0,
                "required": True,
                "scale": 1.0,
                "semantic_role": "Z_COORDINATE",
            },
        ],
        "quantization_policy": "EXACT_TYPED_VALUE",
        "row_order": "ORIGINAL_INDEX_ASCENDING",
        "spec_id": spec_id,
        "spec_schema_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "units": "m",
    }
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _default_digest_kwargs() -> dict[str, Any]:
    return {
        "schema_version": source.FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        "source_model": FrvSourceModel.SINGLE_FILE_ONLY,
        "coordinate_conversion_policy": FrvSourceCoordinateConversionPolicy.FORBID,
        "invalid_point_policy": FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        "canonical_content_policy": FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        "pass_through_dimensions": (),
    }


def test_frvsrcfg1_default_golden_digest() -> None:
    assert derive_frv_source_provider_config_digest(**_default_digest_kwargs()) == (
        "2028bcc1c2b342af5f867c7b49c5cfd1a9e0eda0a1d60effef58158e94a50846"
    )


def test_frvsrcfg1_matches_config_digest_and_is_deterministic() -> None:
    config = _config(pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,))
    expected = derive_frv_source_provider_config_digest(
        schema_version=config.schema_version,
        source_model=config.source_model,
        coordinate_conversion_policy=config.coordinate_conversion_policy,
        invalid_point_policy=config.invalid_point_policy,
        canonical_content_policy=config.canonical_content_policy,
        pass_through_dimensions=config.pass_through_dimensions,
    )
    assert config.provider_config_digest == expected
    assert derive_frv_source_provider_config_digest(
        schema_version=config.schema_version,
        source_model=config.source_model,
        coordinate_conversion_policy=config.coordinate_conversion_policy,
        invalid_point_policy=config.invalid_point_policy,
        canonical_content_policy=config.canonical_content_policy,
        pass_through_dimensions=config.pass_through_dimensions,
    ) == expected


def test_frvsrcfg1_field_sensitivity() -> None:
    base = derive_frv_source_provider_config_digest(**_default_digest_kwargs())
    kwargs = _default_digest_kwargs()
    kwargs["schema_version"] = "frv-source-provider-config-v2"
    assert derive_frv_source_provider_config_digest(**kwargs) != base
    kwargs = _default_digest_kwargs()
    kwargs["pass_through_dimensions"] = (
        FrvSourcePassThroughDimension.CLASSIFICATION,
    )
    assert derive_frv_source_provider_config_digest(**kwargs) != base
    kwargs = _default_digest_kwargs()
    kwargs["source_model"] = source.FrvSourceModel.SINGLE_FILE_ONLY
    assert derive_frv_source_provider_config_digest(**kwargs) == base


def test_frvsrcfg1_rejects_bad_types() -> None:
    with pytest.raises(TypeError, match="schema_version"):
        derive_frv_source_provider_config_digest(
            **_default_digest_kwargs(), schema_version=b"v"
        )
    with pytest.raises(ValueError, match="source_model"):
        derive_frv_source_provider_config_digest(
            **{**_default_digest_kwargs(), "source_model": "single_file_only"}
        )


def test_config_rejects_format_valid_but_wrong_digest() -> None:
    with pytest.raises(ValueError, match="FRVSRCFG1"):
        _config(provider_config_digest="b" * 64)


# ---------------------------------------------------------------------------
# 11. C2: FRVPCD1 streaming point-content digest
# ---------------------------------------------------------------------------


def test_frvpcd1_golden_vector() -> None:
    spec = _exact_canonical_spec()
    chunk = _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1])
    digest = derive_frv_source_point_content_digest(
        [chunk], canonical_point_content_spec=spec
    )
    spec_header = _expected_spec_header_bytes()
    expected = hashlib.sha256()
    expected.update(b"FRVPCD1\0")
    expected.update(struct.pack("<Q", len(spec_header)))
    expected.update(spec_header)
    expected.update(struct.pack("<qddd", 0, 1.0, 2.0, 3.0))
    expected.update(struct.pack("<qddd", 1, 4.0, 5.0, 6.0))
    assert digest == expected.hexdigest()


def test_frvpcd1_chunk_boundary_invariance() -> None:
    spec = _exact_canonical_spec()
    points = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    single = _chunk_with_points(points, [0, 1, 2])
    d_single = derive_frv_source_point_content_digest([single], canonical_point_content_spec=spec)
    first = _chunk_with_points(points[:2], [0, 1], chunk_id=0, start_index=0, is_last=False)
    second = _chunk_with_points(points[2:], [2], chunk_id=1, start_index=2, is_last=True)
    d_multi = derive_frv_source_point_content_digest(
        [first, second], canonical_point_content_spec=spec
    )
    assert d_single == d_multi


def test_frvpcd1_empty_iterable() -> None:
    spec = _exact_canonical_spec()
    digest = derive_frv_source_point_content_digest([], canonical_point_content_spec=spec)
    spec_header = _expected_spec_header_bytes()
    expected = hashlib.sha256()
    expected.update(b"FRVPCD1\0")
    expected.update(struct.pack("<Q", len(spec_header)))
    expected.update(spec_header)
    assert digest == expected.hexdigest()


def test_frvpcd1_changes_alter_digest() -> None:
    spec = _exact_canonical_spec()
    base = _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1])
    d_base = derive_frv_source_point_content_digest([base], canonical_point_content_spec=spec)
    d_xyz = derive_frv_source_point_content_digest(
        [_chunk_with_points([[1.0, 2.0, 9.0], [4.0, 5.0, 6.0]], [0, 1])],
        canonical_point_content_spec=spec,
    )
    assert d_xyz != d_base
    d_order = derive_frv_source_point_content_digest(
        [_chunk_with_points([[4.0, 5.0, 6.0], [1.0, 2.0, 3.0]], [0, 1])],
        canonical_point_content_spec=spec,
    )
    assert d_order != d_base
    d_count = derive_frv_source_point_content_digest(
        [_chunk_with_points([[1.0, 2.0, 3.0]], [0])],
        canonical_point_content_spec=spec,
    )
    assert d_count != d_base


def test_frvpcd1_rejects_non_exact_spec() -> None:
    with pytest.raises(ValueError, match="exactly four fields"):
        derive_frv_source_point_content_digest([], canonical_point_content_spec=_canonical_spec())


def test_frvpcd1_stream_continuity() -> None:
    spec = _exact_canonical_spec()
    with pytest.raises(ValueError, match="chunk_id"):
        derive_frv_source_point_content_digest(
            [_chunk_with_points([[1.0, 2.0, 3.0]], [0], chunk_id=1, start_index=0)],
            canonical_point_content_spec=spec,
        )
    with pytest.raises(ValueError, match="start_index"):
        derive_frv_source_point_content_digest(
            [_chunk_with_points([[1.0, 2.0, 3.0]], [0], chunk_id=0, start_index=5)],
            canonical_point_content_spec=spec,
        )
    first = _chunk_with_points([[1.0, 2.0, 3.0]], [0], chunk_id=0, start_index=0, is_last=False)
    bad_id = _chunk_with_points([[4.0, 5.0, 6.0]], [1], chunk_id=2, start_index=1, is_last=True)
    with pytest.raises(ValueError, match="chunk_id"):
        derive_frv_source_point_content_digest([first, bad_id], canonical_point_content_spec=spec)
    bad_start = _chunk_with_points([[4.0, 5.0, 6.0]], [1], chunk_id=1, start_index=3, is_last=True)
    with pytest.raises(ValueError, match="start_index"):
        derive_frv_source_point_content_digest(
            [first, bad_start], canonical_point_content_spec=spec
        )
    mixed = _chunk_with_points(
        [[4.0, 5.0, 6.0]], [1], source_id="other", chunk_id=1, start_index=1, is_last=True
    )
    with pytest.raises(ValueError, match="source_id"):
        derive_frv_source_point_content_digest([first, mixed], canonical_point_content_spec=spec)
    final_chunk = _chunk_with_points(
        [[1.0, 2.0, 3.0]], [0], chunk_id=0, start_index=0, is_last=True
    )
    more_after_final = _chunk_with_points(
        [[4.0, 5.0, 6.0]], [1], chunk_id=1, start_index=1, is_last=True
    )
    with pytest.raises(ValueError, match="after the final"):
        derive_frv_source_point_content_digest(
            [final_chunk, more_after_final], canonical_point_content_spec=spec
        )
    no_final = _chunk_with_points(
        [[1.0, 2.0, 3.0]], [0], chunk_id=0, start_index=0, is_last=False
    )
    with pytest.raises(ValueError, match="must end with an is_last"):
        derive_frv_source_point_content_digest([no_final], canonical_point_content_spec=spec)
    with pytest.raises(TypeError, match="FrvSourceChunk"):
        derive_frv_source_point_content_digest(["not-a-chunk"], canonical_point_content_spec=spec)


# ---------------------------------------------------------------------------
# 12. C2: FRVA2IDX1 streaming original-index digest
# ---------------------------------------------------------------------------


def test_frva2idx1_golden_vector_and_interop() -> None:
    chunk = _chunk_with_points(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [0, 1, 2]
    )
    digest = derive_frv_source_original_index_digest(
        [chunk], source_content_identity_digest=SHA
    )
    expected = hashlib.sha256()
    expected.update(b"FRVA2IDX1\0")
    expected.update(b"ORIGINAL_INDEX")
    expected.update(b"\0")
    expected.update(SHA.encode("utf-8"))
    expected.update(b"\0")
    expected.update(FrvA2DomainKind.FULL_SOURCE.value.encode("utf-8"))
    expected.update(b"\0")
    expected.update(struct.pack("<Q", 3))
    for index in range(3):
        expected.update(struct.pack("<Q", index))
    assert digest == expected.hexdigest()
    assert digest == frv_a2_index_domain_digest(
        "ORIGINAL_INDEX", SHA, FrvA2DomainKind.FULL_SOURCE, tuple(range(3))
    )


def test_frva2idx1_chunk_boundary_invariance() -> None:
    first = _chunk_with_points(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1], chunk_id=0, start_index=0, is_last=False
    )
    second = _chunk_with_points([[7.0, 8.0, 9.0]], [2], chunk_id=1, start_index=2, is_last=True)
    single = _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], [0, 1, 2])
    assert derive_frv_source_original_index_digest(
        [first, second], source_content_identity_digest=SHA
    ) == (
        derive_frv_source_original_index_digest([single], source_content_identity_digest=SHA)
    )


def test_frva2idx1_empty_and_changes() -> None:
    empty = derive_frv_source_original_index_digest([], source_content_identity_digest=SHA)
    expected = hashlib.sha256()
    expected.update(b"FRVA2IDX1\0")
    expected.update(b"ORIGINAL_INDEX")
    expected.update(b"\0")
    expected.update(SHA.encode("utf-8"))
    expected.update(b"\0")
    expected.update(FrvA2DomainKind.FULL_SOURCE.value.encode("utf-8"))
    expected.update(b"\0")
    expected.update(struct.pack("<Q", 0))
    assert empty == expected.hexdigest()

    one = _chunk_with_points([[1.0, 2.0, 3.0]], [0])
    d_one = derive_frv_source_original_index_digest([one], source_content_identity_digest=SHA)
    two = _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1])
    d_two = derive_frv_source_original_index_digest([two], source_content_identity_digest=SHA)
    assert d_two != d_one


def test_frva2idx1_rejects_non_global_sequence() -> None:
    bad = _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 5])
    with pytest.raises(ValueError, match="0..N-1"):
        derive_frv_source_original_index_digest([bad], source_content_identity_digest=SHA)


def test_frva2idx1_rejects_bad_source_digest() -> None:
    chunk = _chunk_with_points([[1.0, 2.0, 3.0]], [0])
    with pytest.raises(TypeError, match="built-in str"):
        derive_frv_source_original_index_digest([chunk], source_content_identity_digest=123)
    with pytest.raises(ValueError, match="SHA-256"):
        derive_frv_source_original_index_digest([chunk], source_content_identity_digest="not-a-sha")


# ---------------------------------------------------------------------------
# 13. C2: full reference validator
# ---------------------------------------------------------------------------


def _valid_reference() -> tuple[FrvSourceProviderReference, FrvSourceProviderConfig]:
    config = _config()
    reference = _reference(
        runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
        source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
        provider_config_digest=config.provider_config_digest,
    )
    return reference, config


def test_reference_validator_success_with_synthetic_absolute_path() -> None:
    reference, config = _valid_reference()
    assert validate_frv_source_provider_reference(
        reference, provider_config=config, expected_configuration_digest=SHA
    ) is None


def test_reference_validator_wrong_role() -> None:
    reference, config = _valid_reference()
    bad = _reference(
        provider_role=FrvA2ProviderRole.HAG,
        runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
        source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
        provider_config_digest=config.provider_config_digest,
    )
    with pytest.raises(ValueError, match="SOURCE_SCAN"):
        validate_frv_source_provider_reference(
            bad, provider_config=config, expected_configuration_digest=SHA
        )


def test_reference_validator_wrong_key() -> None:
    reference, config = _valid_reference()
    bad = _reference(
        provider_key="not_source_scan",
        runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
        source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
        provider_config_digest=config.provider_config_digest,
    )
    with pytest.raises(ValueError, match="source_scan"):
        validate_frv_source_provider_reference(
            bad, provider_config=config, expected_configuration_digest=SHA
        )


def test_reference_validator_component_layout() -> None:
    reference, config = _valid_reference()
    with pytest.raises(ValueError, match="exactly one"):
        validate_frv_source_provider_reference(
            _reference(
                artifact_components=(),
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )
    extra = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256="b" * 64,
        size_bytes=456,
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_frv_source_provider_reference(
            _reference(
                artifact_components=(_component(), extra),
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )
    bad_kind = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.MANIFEST_JSON,
        sha256=SHA,
        size_bytes=123,
    )
    with pytest.raises(ValueError, match="SOURCE_FILE"):
        validate_frv_source_provider_reference(
            _reference(
                artifact_components=(bad_kind,),
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )
    bad_sha = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256="b" * 64,
        size_bytes=123,
    )
    with pytest.raises(ValueError, match="sha256"):
        validate_frv_source_provider_reference(
            _reference(
                artifact_components=(bad_sha,),
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )


def test_reference_validator_wrong_artifact_reference_id() -> None:
    reference, config = _valid_reference()
    wrong_id = "frv-a2-ref-v1|source_scan|source_scan|source_file,source_file," + "b" * 64 + ",123"
    bad = _reference(
        artifact_reference_id=wrong_id,
        runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
        source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
        provider_config_digest=config.provider_config_digest,
    )
    with pytest.raises(ValueError, match="artifact_reference_id"):
        validate_frv_source_provider_reference(
            bad, provider_config=config, expected_configuration_digest=SHA
        )


def test_reference_validator_path_binding() -> None:
    reference, config = _valid_reference()
    with pytest.raises(ValueError, match="runtime_access_path"):
        validate_frv_source_provider_reference(
            _reference(
                runtime_access_path=_SYNTHETIC_WINDOWS_OTHER_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )
    with pytest.raises(ValueError, match="absolute"):
        validate_frv_source_provider_reference(
            _reference(
                runtime_access_path=_FIXTURE_SOURCE_PATH,
                source_identity=_source_identity(path=_FIXTURE_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )


def test_reference_validator_digest_binding() -> None:
    reference, config = _valid_reference()
    with pytest.raises(ValueError, match="configuration_digest"):
        validate_frv_source_provider_reference(
            reference, provider_config=config, expected_configuration_digest="b" * 64
        )
    with pytest.raises(ValueError, match="provider_config_digest"):
        validate_frv_source_provider_reference(
            _reference(
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest="c" * 64,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )


def test_reference_validator_version_binding() -> None:
    reference, config = _valid_reference()
    for name in ("interface_version", "schema_version", "algorithm_version"):
        with pytest.raises(ValueError, match=name):
            validate_frv_source_provider_reference(
                _reference(
                    **{name: "wrong-version"},
                    runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                    source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                    provider_config_digest=config.provider_config_digest,
                ),
                provider_config=config,
                expected_configuration_digest=SHA,
            )


def test_reference_validator_canonical_spec_binding() -> None:
    reference, config = _valid_reference()
    with pytest.raises(ValueError, match="must be null"):
        validate_frv_source_provider_reference(
            _reference(
                canonical_point_content_spec=_exact_canonical_spec(),
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=_source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )
    spec = _exact_canonical_spec()
    source_with_canonical = _source_identity(
        path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT,
        canonical_digest="b" * 64,
        canonical_spec_digest=spec.spec_digest,
        canonical_spec_version=frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
    )
    with pytest.raises(ValueError, match="is required"):
        validate_frv_source_provider_reference(
            _reference(
                runtime_access_path=_SYNTHETIC_WINDOWS_SOURCE_PATH,
                source_identity=source_with_canonical,
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )


def test_reference_validator_type_bypass() -> None:
    config = _config()
    with pytest.raises(TypeError, match="reference"):
        validate_frv_source_provider_reference(
            "not-a-reference", provider_config=config, expected_configuration_digest=SHA
        )
    reference, _config_used = _valid_reference()
    with pytest.raises(TypeError, match="provider_config"):
        validate_frv_source_provider_reference(
            reference, provider_config="not-a-config", expected_configuration_digest=SHA
        )
    with pytest.raises(TypeError, match="expected_configuration_digest"):
        validate_frv_source_provider_reference(
            reference, provider_config=config, expected_configuration_digest=123
        )


# ---------------------------------------------------------------------------
# 14. C2: public exports and stage boundary
# ---------------------------------------------------------------------------


def test_c2_public_exports_present() -> None:
    for name in (
        "derive_frv_source_provider_config_digest",
        "derive_frv_source_point_content_digest",
        "derive_frv_source_original_index_digest",
        "validate_frv_source_provider_reference",
    ):
        assert hasattr(contracts, name)
    assert not hasattr(contracts, "_frv_source_windows_path_comparison_key")


def test_comparison_key_private_behaviour() -> None:
    assert source._frv_source_windows_path_comparison_key(
        r"C:\Data\src.las"
    ) == source._frv_source_windows_path_comparison_key("c:/data/src.las")
    with pytest.raises(ValueError, match="absolute"):
        source._frv_source_windows_path_comparison_key("fixtures/source.las")
    with pytest.raises(ValueError, match="dot segments"):
        source._frv_source_windows_path_comparison_key(r"C:\data\..\src.las")


# ---------------------------------------------------------------------------
# 15. C2: Windows lexical comparison key path rules
# ---------------------------------------------------------------------------


def test_path_key_drive_legal_forms_and_equivalence() -> None:
    key = source._frv_source_windows_path_comparison_key
    assert key(r"C:\Data\Src.las") == "C:\\data\\src.las"
    assert key(r"C:\__frv_source_contract_test__\source.las") == key(
        "C:/__frv_source_contract_test__/source.las"
    )
    assert key(r"C:\__frv_source_contract_test__\source.las") == key(
        r"c:\__frv_source_contract_test__\source.las"
    )


def test_path_key_unicode_casefold_equivalence() -> None:
    key = source._frv_source_windows_path_comparison_key
    assert key(r"C:\Straße\Datei.las") == key(r"C:\STRASSE\datei.las")


def test_path_key_unc_legal_forms_and_equivalence() -> None:
    key = source._frv_source_windows_path_comparison_key
    assert key(r"\\Server\Share\Data.las") == "\\\\server\\share\\data.las"
    assert key(r"\\example.invalid\frv-source-test\source.las") == key(
        "//EXAMPLE.INVALID/FRV-SOURCE-TEST/SOURCE.LAS"
    )
    assert key(r"\\example.invalid\frv-source-test\source.las") == key(
        r"\\EXAMPLE.INVALID\frv-source-test\source.las"
    )


def test_path_key_rejects_relative_rooted_and_drive_relative() -> None:
    key = source._frv_source_windows_path_comparison_key
    for bad in (
        "",
        "source.las",
        "folder/source.las",
        r"folder\source.las",
        r"\folder\source.las",
        "/folder/source.las",
        r"C:folder\source.las",
        "C:",
    ):
        with pytest.raises(ValueError):
            key(bad)


def test_path_key_rejects_device_namespaces() -> None:
    key = source._frv_source_windows_path_comparison_key
    for bad in (
        r"\\?\C:\folder\source.las",
        r"\\.\C:\folder\source.las",
        r"\\?\UNC\server\share\source.las",
        r"\\.\pipe\name",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\source.las",
    ):
        with pytest.raises(ValueError, match="device"):
            key(bad)


def test_path_key_rejects_illegal_unc_server_and_share() -> None:
    key = source._frv_source_windows_path_comparison_key
    for bad in (
        r"\\..\share\file.las",
        r"\\server.\share\file.las",
        r"\\server \share\file.las",
        r"\\server:1\share\file.las",
    ):
        with pytest.raises(ValueError):
            key(bad)
    for bad in (
        r"\\server\.\file.las",
        r"\\server\..\file.las",
        r"\\server\share.\file.las",
        r"\\server\share \file.las",
        r"\\server\share:1\file.las",
    ):
        with pytest.raises(ValueError):
            key(bad)
    with pytest.raises(ValueError, match="server and share"):
        key(r"\\\share\file.las")
    with pytest.raises(ValueError, match="server and share"):
        key(r"\\server\\file.las")


def test_path_key_rejects_illegal_components_drive_and_unc() -> None:
    key = source._frv_source_windows_path_comparison_key
    for bad in (
        r"C:\folder\.\source.las",
        r"C:\folder\..\source.las",
        r"C:\folder\name.\source.las",
        r"C:\folder\name \source.las",
        r"C:\folder\file.txt:stream",
        r"C:\folder\folder:name",
        r"C:\folder\nested\..\source.las",
        r"\\server\share\.\source.las",
        r"\\server\share\..\source.las",
        r"\\server\share\name.\source.las",
        r"\\server\share\name \source.las",
        r"\\server\share\file.txt:stream",
        r"\\server\share\folder:name",
    ):
        with pytest.raises(ValueError):
            key(bad)


def test_path_key_rejects_nul_anywhere() -> None:
    key = source._frv_source_windows_path_comparison_key
    for bad in (
        "\x00C:\\folder\\source.las",
        "C:\\fo\x00lder\\source.las",
        "C:\\folder\\sour\x00ce.las",
        "\\\\ser\x00ver\\share\\file.las",
        "\\\\server\\share\\na\x00me.las",
    ):
        with pytest.raises(ValueError, match="NUL"):
            key(bad)


def test_path_key_rejects_non_builtin_str_and_bytes() -> None:
    key = source._frv_source_windows_path_comparison_key

    class _StrSubclass(str):
        pass

    with pytest.raises(TypeError, match="built-in str"):
        key(_StrSubclass(r"C:\folder\source.las"))
    with pytest.raises(TypeError, match="built-in str"):
        key(b"C:\\folder\\source.las")


# ---------------------------------------------------------------------------
# 16. C2: reference validator path binding rules
# ---------------------------------------------------------------------------


def test_reference_validator_success_with_synthetic_unc_path() -> None:
    config = _config()
    unc_text = r"\\example.invalid\frv-source-test\source.las"
    reference = _reference(
        runtime_access_path=Path(unc_text),
        source_identity=_source_identity(path=unc_text),
        provider_config_digest=config.provider_config_digest,
    )
    assert validate_frv_source_provider_reference(
        reference, provider_config=config, expected_configuration_digest=SHA
    ) is None


def test_reference_validator_lexical_equivalent_paths_match() -> None:
    config = _config()
    reference = _reference(
        runtime_access_path=Path(r"C:\DATA\SRC.LAS"),
        source_identity=_source_identity(path=r"C:\data\src.las"),
        provider_config_digest=config.provider_config_digest,
    )
    assert validate_frv_source_provider_reference(
        reference, provider_config=config, expected_configuration_digest=SHA
    ) is None


def test_reference_validator_rejects_illegal_runtime_paths() -> None:
    config = _config()
    source_identity = _source_identity(path=_SYNTHETIC_WINDOWS_SOURCE_PATH_TEXT)
    cases = (
        (Path(r"\\?\C:\folder\source.las"), "device"),
        (Path(r"\\..\share\file.las"), "dot segments"),
        (Path(r"C:\__frv_source_contract_test__\name.\source.las"), "trailing"),
        (Path(r"C:\__frv_source_contract_test__\source.las:stream"), "colon"),
        (Path("C:\\__frv_source_contract_test__\\source\x00.las"), "NUL"),
    )
    for runtime_path, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_frv_source_provider_reference(
                _reference(
                    runtime_access_path=runtime_path,
                    source_identity=source_identity,
                    provider_config_digest=config.provider_config_digest,
                ),
                provider_config=config,
                expected_configuration_digest=SHA,
            )


def test_reference_validator_rejects_illegal_resolved_path() -> None:
    config = _config()
    runtime_path = _SYNTHETIC_WINDOWS_SOURCE_PATH
    with pytest.raises(ValueError, match="dot segments"):
        validate_frv_source_provider_reference(
            _reference(
                runtime_access_path=runtime_path,
                source_identity=_source_identity(
                    path=r"C:\__frv_source_contract_test__\..\source.las"
                ),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )
    with pytest.raises(ValueError, match="colon"):
        validate_frv_source_provider_reference(
            _reference(
                runtime_access_path=runtime_path,
                source_identity=_source_identity(
                    path=r"C:\__frv_source_contract_test__\source.las:stream"
                ),
                provider_config_digest=config.provider_config_digest,
            ),
            provider_config=config,
            expected_configuration_digest=SHA,
        )


# ---------------------------------------------------------------------------
# 17. C2: generator single-pass and no-full-source-materialization
# ---------------------------------------------------------------------------


class _SinglePassChunkIterable:
    """Iterable asserting it is consumed exactly once without len/getitem."""

    def __init__(self, chunks: list[FrvSourceChunk]) -> None:
        self._chunks = tuple(chunks)
        self.iteration_count = 0

    def __iter__(self):
        self.iteration_count += 1
        if self.iteration_count != 1:
            raise AssertionError("chunk iterable consumed more than once")
        yield from self._chunks

    def __len__(self) -> int:
        raise AssertionError("chunk iterable must not be len()-ed")

    def __getitem__(self, index: int) -> FrvSourceChunk:
        raise AssertionError("chunk iterable must not be subscripted")


class _StreamingChunkIterable:
    """Lazily built chunks that verify earlier chunk arrays are released.

    Each chunk is constructed on demand; only weak references to previously
    yielded arrays are kept.  When a chunk two steps back is requested, its
    arrays must already have been released, which fails for any helper that
    materializes the full source (for example via ``list(chunks)``).  The
    iterable also records single-use consumption and natural exhaustion so a
    helper that skips chunks (or the whole iterable) cannot pass.
    """

    _POINTS_PER_CHUNK = 2
    _CHUNK_COUNT = 5

    def __init__(self) -> None:
        self._index = 0
        self.iteration_count = 0
        self.exhausted = False
        self._array_refs: list[tuple[Any, Any]] = []

    def _make_chunk(self) -> FrvSourceChunk:
        start = self._index * self._POINTS_PER_CHUNK
        xyz = _readonly(
            np.ascontiguousarray(
                np.array(
                    [[float(start), 0.0, 0.0], [float(start + 1), 1.0, 1.0]],
                    dtype="<f8",
                )
            )
        )
        original = _readonly(np.ascontiguousarray(np.array([start, start + 1], dtype="<i8")))
        return FrvSourceChunk(
            source_id="scene-1",
            chunk_id=self._index,
            start_index=start,
            point_count=self._POINTS_PER_CHUNK,
            xyz=xyz,
            original_index=original,
            attributes=MappingProxyType({}),
            is_last=self._index == self._CHUNK_COUNT - 1,
        )

    def __iter__(self):
        self.iteration_count += 1
        if self.iteration_count != 1:
            raise AssertionError("chunk iterable consumed more than once")
        while self._index < self._CHUNK_COUNT:
            chunk = self._make_chunk()
            self._array_refs.append((weakref.ref(chunk.xyz), weakref.ref(chunk.original_index)))
            if len(self._array_refs) > 2:
                gc.collect()
                old_xyz, old_index = self._array_refs[-3]
                if old_xyz() is not None or old_index() is not None:
                    raise AssertionError("digest helper retained earlier chunk arrays")
            self._index += 1
            yield chunk
        self.exhausted = True

    def __len__(self) -> int:
        raise AssertionError("chunk iterable must not be len()-ed")

    def __getitem__(self, index: int) -> FrvSourceChunk:
        raise AssertionError("chunk iterable must not be subscripted")


def _streaming_reference_chunks() -> list[FrvSourceChunk]:
    """Independently build the same 5-chunk stream the streaming iterable yields.

    The chunks are plain, freshly constructed objects (never collected from the
    guarded iterable) so they can be used to compute an expected full digest.
    """

    chunks: list[FrvSourceChunk] = []
    per_chunk = _StreamingChunkIterable._POINTS_PER_CHUNK
    for index in range(_StreamingChunkIterable._CHUNK_COUNT):
        start = index * per_chunk
        chunks.append(
            _chunk_with_points(
                [[float(start), 0.0, 0.0], [float(start + 1), 1.0, 1.0]],
                [start, start + 1],
                chunk_id=index,
                start_index=start,
                is_last=index == _StreamingChunkIterable._CHUNK_COUNT - 1,
            )
        )
    return chunks


def test_frvpcd1_generator_single_pass() -> None:
    spec = _exact_canonical_spec()
    chunks = _SinglePassChunkIterable(
        [
            _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1], is_last=False),
            _chunk_with_points([[7.0, 8.0, 9.0]], [2], chunk_id=1, start_index=2, is_last=True),
        ]
    )
    digest = derive_frv_source_point_content_digest(chunks, canonical_point_content_spec=spec)
    assert chunks.iteration_count == 1
    assert digest == derive_frv_source_point_content_digest(
        chunks._chunks, canonical_point_content_spec=spec
    )


def test_frva2idx1_generator_single_pass() -> None:
    chunks = _SinglePassChunkIterable(
        [
            _chunk_with_points([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [0, 1], is_last=False),
            _chunk_with_points([[7.0, 8.0, 9.0]], [2], chunk_id=1, start_index=2, is_last=True),
        ]
    )
    digest = derive_frv_source_original_index_digest(chunks, source_content_identity_digest=SHA)
    assert chunks.iteration_count == 1
    assert digest == derive_frv_source_original_index_digest(
        chunks._chunks, source_content_identity_digest=SHA
    )


def test_frvpcd1_no_full_source_materialization() -> None:
    spec = _exact_canonical_spec()
    chunks = _StreamingChunkIterable()
    actual = derive_frv_source_point_content_digest(chunks, canonical_point_content_spec=spec)

    reference = _streaming_reference_chunks()
    expected = derive_frv_source_point_content_digest(reference, canonical_point_content_spec=spec)
    assert actual == expected
    assert chunks.iteration_count == 1
    assert chunks._index == _StreamingChunkIterable._CHUNK_COUNT
    assert chunks.exhausted is True

    # The final chunk must influence the digest: changing only the last point's
    # z coordinate of the last chunk must change the FRVPCD1 digest.
    last = reference[-1]
    modified_xyz = last.xyz.tolist()
    modified_xyz[-1][2] = modified_xyz[-1][2] + 1.0
    modified_last = _chunk_with_points(
        modified_xyz,
        last.original_index.tolist(),
        chunk_id=last.chunk_id,
        start_index=last.start_index,
        is_last=last.is_last,
    )
    assert derive_frv_source_point_content_digest(
        reference[:-1] + [modified_last], canonical_point_content_spec=spec
    ) != expected


def test_frva2idx1_no_full_source_materialization() -> None:
    chunks = _StreamingChunkIterable()
    actual = derive_frv_source_original_index_digest(chunks, source_content_identity_digest=SHA)

    reference = _streaming_reference_chunks()
    expected = derive_frv_source_original_index_digest(
        reference, source_content_identity_digest=SHA
    )
    assert actual == expected
    assert chunks.iteration_count == 1
    assert chunks._index == _StreamingChunkIterable._CHUNK_COUNT
    assert chunks.exhausted is True

    # The final chunk must influence the result: FRVA2IDX1 hashes the global
    # 0..N-1 sequence derived from the full stream, so a non-global
    # original_index in the last chunk must be rejected.  A helper that skips
    # the last chunk's index data would not reject it.
    last = reference[-1]
    bad_original = last.original_index.tolist()
    bad_original[-1] += 1
    bad_last = _chunk_with_points(
        last.xyz.tolist(),
        bad_original,
        chunk_id=last.chunk_id,
        start_index=last.start_index,
        is_last=last.is_last,
    )
    with pytest.raises(ValueError, match="0..N-1"):
        derive_frv_source_original_index_digest(
            reference[:-1] + [bad_last], source_content_identity_digest=SHA
        )


# ---------------------------------------------------------------------------
# 18. C3-A: load error and single-use load session protocol
# ---------------------------------------------------------------------------


def _failure_record(
    *,
    phase: FrvA2FailurePhase = FrvA2FailurePhase.STRUCTURE_INTEGRITY,
    reason_flags: int = 0,
) -> FrvA2FailureRecord:
    role = FrvA2ProviderRole.SOURCE_SCAN
    key = "source_scan"
    field = "manifest"
    error = FrvErrorCode.FRV_PROVIDER_INVALID
    detail = f"{phase.value}:{role.value}:{key}:{field}:{error.value}:{reason_flags}"
    return FrvA2FailureRecord(phase, role, key, field, error, reason_flags, detail)


def _quality_flag_known_mask() -> int:
    known = 0
    for member in FrvQualityFlag.__members__.values():
        known |= int(member.value)
    return known


def _quality_flag_next_unknown_bit() -> int:
    return max(int(member.value) for member in FrvQualityFlag.__members__.values()) * 2


def test_source_load_error_normal_construction() -> None:
    record = _failure_record()
    error = FrvSourceLoadError(failure_record=record, quality_flags=0b101)
    assert isinstance(error, Exception)
    assert error.failure_record is record
    assert error.quality_flags == 0b101
    assert str(error) == record.detail
    assert error.args == (record.detail,)


def test_source_load_error_constructor_signature() -> None:
    parameters = list(inspect.signature(FrvSourceLoadError.__init__).parameters.values())
    assert [parameter.name for parameter in parameters] == [
        "self",
        "failure_record",
        "quality_flags",
    ]
    assert parameters[1].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters[2].kind is inspect.Parameter.KEYWORD_ONLY


def test_source_load_error_rejects_positional_construction() -> None:
    record = _failure_record()
    with pytest.raises(TypeError):
        FrvSourceLoadError(record, 0)  # type: ignore[misc]


def test_source_load_error_rejects_wrong_failure_record_type() -> None:
    with pytest.raises(TypeError, match="failure_record"):
        FrvSourceLoadError(failure_record=None, quality_flags=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="failure_record"):
        FrvSourceLoadError(failure_record={}, quality_flags=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="failure_record"):
        FrvSourceLoadError(failure_record="record", quality_flags=0)  # type: ignore[arg-type]


def test_source_load_error_rejects_failure_record_duck_type() -> None:
    class _FailureDuck:
        phase = FrvA2FailurePhase.STRUCTURE_INTEGRITY
        provider_role = FrvA2ProviderRole.SOURCE_SCAN
        provider_key = "source_scan"
        field_name = "manifest"
        error_code = FrvErrorCode.FRV_PROVIDER_INVALID
        reason_flags = 0
        detail = "duck"

    with pytest.raises(TypeError, match="failure_record"):
        FrvSourceLoadError(failure_record=_FailureDuck(), quality_flags=0)  # type: ignore[arg-type]


def test_source_load_error_rejects_failure_record_subclass() -> None:
    record = _failure_record()

    class _FailureSubclass(FrvA2FailureRecord):
        pass

    subclass_instance = _FailureSubclass.__new__(_FailureSubclass)
    for name in FrvA2FailureRecord.__dataclass_fields__:
        object.__setattr__(subclass_instance, name, getattr(record, name))
    with pytest.raises(TypeError, match="failure_record"):
        FrvSourceLoadError(failure_record=subclass_instance, quality_flags=0)  # type: ignore[arg-type]


def test_source_load_error_rejects_invalid_failure_record() -> None:
    raw = FrvA2FailureRecord.__new__(FrvA2FailureRecord)
    for name in FrvA2FailureRecord.__dataclass_fields__:
        object.__setattr__(raw, name, getattr(_failure_record(), name))
    object.__setattr__(raw, "reason_flags", -1)
    with pytest.raises(ValueError, match="reason_flags"):
        FrvSourceLoadError(failure_record=raw, quality_flags=0)


def test_source_load_error_delegates_to_failure_record_validator() -> None:
    record = _failure_record()
    with mock.patch(
        "core.contracts.frv_source.validate_frv_a2_failure_record",
        wraps=validate_frv_a2_failure_record,
    ) as mocked:
        error = FrvSourceLoadError(failure_record=record, quality_flags=0)
        mocked.assert_called_once_with(record)
        assert error.failure_record is record


def test_source_load_error_rejects_wrong_quality_flags_type() -> None:
    record = _failure_record()
    for bad in (None, "0", 1.0, [0], 0.5):
        with pytest.raises(TypeError, match="quality_flags"):
            FrvSourceLoadError(failure_record=record, quality_flags=bad)  # type: ignore[arg-type]


def test_source_load_error_rejects_bool_quality_flags() -> None:
    record = _failure_record()
    with pytest.raises(TypeError, match="quality_flags"):
        FrvSourceLoadError(failure_record=record, quality_flags=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="quality_flags"):
        FrvSourceLoadError(failure_record=record, quality_flags=False)  # type: ignore[arg-type]


def test_source_load_error_rejects_int_subclass_quality_flags() -> None:
    record = _failure_record()

    class _IntSubclass(int):
        pass

    with pytest.raises(TypeError, match="quality_flags"):
        FrvSourceLoadError(  # type: ignore[arg-type]
            failure_record=record, quality_flags=_IntSubclass(0)
        )


def test_source_load_error_rejects_negative_quality_flags() -> None:
    record = _failure_record()
    with pytest.raises(ValueError, match="nonnegative"):
        FrvSourceLoadError(failure_record=record, quality_flags=-1)


def test_source_load_error_quality_flags_zero_accepted() -> None:
    record = _failure_record()
    error = FrvSourceLoadError(failure_record=record, quality_flags=0)
    assert error.quality_flags == 0


def test_source_load_error_quality_flags_single_known_bit_accepted() -> None:
    record = _failure_record()
    flags = int(FrvQualityFlag.ORIGINAL_INDEX_INVALID)
    error = FrvSourceLoadError(failure_record=record, quality_flags=flags)
    assert error.quality_flags == flags


def test_source_load_error_quality_flags_combined_known_bits_accepted() -> None:
    record = _failure_record()
    flags = int(FrvQualityFlag.ORIGINAL_INDEX_INVALID) | int(
        FrvQualityFlag.STREAM_INCOMPLETE
    )
    error = FrvSourceLoadError(failure_record=record, quality_flags=flags)
    assert error.quality_flags == flags


def test_source_load_error_quality_flags_rejects_unknown_bit() -> None:
    record = _failure_record()
    unknown = _quality_flag_next_unknown_bit()
    assert unknown & ~_quality_flag_known_mask()
    with pytest.raises(ValueError, match="unknown flag bit"):
        FrvSourceLoadError(failure_record=record, quality_flags=unknown)


def test_source_load_error_quality_flags_rejects_known_plus_unknown_bit() -> None:
    record = _failure_record()
    mixed = int(FrvQualityFlag.ORIGINAL_INDEX_INVALID) | _quality_flag_next_unknown_bit()
    with pytest.raises(ValueError, match="unknown flag bit"):
        FrvSourceLoadError(failure_record=record, quality_flags=mixed)


def test_source_load_error_delegates_to_flag_mask_validator() -> None:
    record = _failure_record()
    flags = int(FrvQualityFlag.ORIGINAL_INDEX_INVALID)
    with mock.patch(
        "core.contracts.frv_source.validate_frv_flag_mask",
        wraps=validate_frv_flag_mask,
    ) as mocked:
        error = FrvSourceLoadError(failure_record=record, quality_flags=flags)
        mocked.assert_called_once_with(flags, FrvQualityFlag, "quality_flags")
        assert error.quality_flags == flags


def test_source_load_error_supports_from_chaining() -> None:
    record = _failure_record()
    raw_error = RuntimeError("raw io detail")
    try:
        raise FrvSourceLoadError(failure_record=record, quality_flags=1) from raw_error
    except FrvSourceLoadError as error:
        assert error.__cause__ is raw_error
        assert str(error) == record.detail
        assert "raw io detail" not in str(error)
        assert error.args == (record.detail,)
        assert not hasattr(error, "raw_exception")
        assert not hasattr(error, "partial_result")


def test_source_load_session_is_protocol() -> None:
    from typing import Protocol

    assert issubclass(FrvSourceLoadSession, Protocol)
    with pytest.raises(TypeError):
        FrvSourceLoadSession()  # type: ignore[call-arg]


def test_source_load_session_not_runtime_checkable() -> None:
    class _Dummy:
        def iter_chunks(self): ...
        def finalize(self): ...
        def close(self): ...
        def __enter__(self): ...
        def __exit__(self, exc_type, exc, traceback): ...

    with pytest.raises(TypeError):
        issubclass(_Dummy, FrvSourceLoadSession)


def test_source_load_session_exact_method_names() -> None:
    methods = {
        name
        for name in vars(FrvSourceLoadSession)
        if not name.startswith("_") or name in ("__enter__", "__exit__")
    }
    assert methods == {"iter_chunks", "finalize", "close", "__enter__", "__exit__"}
    for forbidden in ("state", "reader", "handle", "path", "result", "failure", "loader"):
        assert not hasattr(FrvSourceLoadSession, forbidden)


def test_source_load_session_method_signatures() -> None:
    assert list(inspect.signature(FrvSourceLoadSession.iter_chunks).parameters) == ["self"]
    assert list(inspect.signature(FrvSourceLoadSession.finalize).parameters) == ["self"]
    assert list(inspect.signature(FrvSourceLoadSession.close).parameters) == ["self"]
    assert list(inspect.signature(FrvSourceLoadSession.__enter__).parameters) == ["self"]
    assert list(inspect.signature(FrvSourceLoadSession.__exit__).parameters) == [
        "self",
        "exc_type",
        "exc",
        "traceback",
    ]


def test_source_load_session_return_annotations() -> None:
    def _annotation_text(method) -> str:
        return str(inspect.signature(method).return_annotation).strip("'\"")

    assert _annotation_text(FrvSourceLoadSession.iter_chunks) == "Iterator[FrvSourceChunk]"
    assert _annotation_text(FrvSourceLoadSession.finalize) == "FrvSourceProviderResult"
    assert _annotation_text(FrvSourceLoadSession.close) == "None"
    assert _annotation_text(FrvSourceLoadSession.__enter__) == "FrvSourceLoadSession"
    assert _annotation_text(FrvSourceLoadSession.__exit__) == "Literal[False]"


def test_c3a_public_exports_present() -> None:
    assert contracts.FrvSourceLoadError is source.FrvSourceLoadError
    assert contracts.FrvSourceLoadSession is source.FrvSourceLoadSession
    assert issubclass(source.FrvSourceLoadError, Exception)
    for name in (
        "FrvSourceProviderProof",
        "FrvSourceProviderResult",
        "validate_finalized_frv_source_provider_result",
        "to_frv_a2_provider_binding_evidence",
        "load_frv_source",
        "_FRV_SOURCE_FINALIZED_CAPABILITY",
        "_create_finalized_frv_source_provider_result",
    ):
        assert not hasattr(contracts, name)
