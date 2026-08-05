"""FRV-A2 P-SOURCE runtime provider session tests (C4/C5).

Synthetic LAS fixtures are generated in the pytest temp dir; no committed
binary fixtures and no network/PDAL/CloudCompare dependency.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

import core.contracts as contracts
import core.contracts.frv as frv
import laspy
import numpy as np
import pytest
from algorithms.io import load_frv_source, source_provider
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2ProviderRole,
)
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourceCanonicalContentPolicy,
    FrvSourceCoordinateConversionPolicy,
    FrvSourceInvalidPointPolicy,
    FrvSourceLoadError,
    FrvSourceModel,
    FrvSourcePassThroughDimension,
    FrvSourceProviderConfig,
    FrvSourceProviderReference,
    derive_frv_source_point_content_digest,
    derive_frv_source_provider_config_digest,
)

from tests.contracts.test_frv_source_contracts import (
    _chunk_with_points,
    _exact_canonical_spec,
)

SHA = "a" * 64


def _load(reference, config, configuration, *, chunk_size: int = 3):
    return load_frv_source(
        reference,
        frv_configuration=configuration,
        provider_config=config,
        chunk_size=chunk_size,
    )


def write_las(path: Path, *, point_count: int = 7, point_format: int = 7) -> None:
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    las = laspy.LasData(header)
    idx = np.arange(point_count)
    las.x = 100.0 + idx.astype(np.float64)
    las.y = 200.0 + idx.astype(np.float64)
    las.z = idx.astype(np.float64)
    dims = tuple(las.point_format.dimension_names)
    if "intensity" in dims:
        las.intensity = (idx + 10).astype(np.uint16)
    if "classification" in dims:
        las.classification = (idx % 5).astype(np.uint8)
    if "point_source_id" in dims:
        las.point_source_id = (idx + 20).astype(np.uint16)
    las.write(path)


def _bounds_for(count: int) -> tuple[list[float], list[float]]:
    return (
        [100.0, 200.0, 0.0],
        [100.0 + count - 1, 200.0 + count - 1, count - 1.0],
    )


def _source_identity(
    path: str,
    file_sha: str,
    count: int,
    bounds: tuple[list[float], list[float]],
    *,
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
    relocation_payload["audit_digest"] = frv.frv_digest(relocation_payload)
    relocation = frv.FrvRelocationAudit.from_dict(relocation_payload)
    content_payload = {
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_id": "scene-1",
        "source_file_sha256": file_sha,
        "canonical_point_content_digest": canonical_digest,
        "canonical_point_content_spec_digest": canonical_spec_digest,
        "canonical_point_content_spec_version": canonical_spec_version,
        "source_point_count": count,
        "source_bounds_min_xyz": list(bounds[0]),
        "source_bounds_max_xyz": list(bounds[1]),
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


def _provider_config(*, pass_through_dimensions=()) -> FrvSourceProviderConfig:
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=pass_through_dimensions,
        provider_config_digest=derive_frv_source_provider_config_digest(
            schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
            source_model=FrvSourceModel.SINGLE_FILE_ONLY,
            coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
            invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
            canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
            pass_through_dimensions=pass_through_dimensions,
        ),
    )


def _configuration() -> frv.FrvConfiguration:
    payload = {
        "schema_version": frv.FRV_CONFIGURATION_SCHEMA_VERSION,
        "configuration_version": "config-1",
        "algorithm_version": "alg-1",
        "decision_rule_version": "rule-1",
        "canonicalization_version": frv._FRV_CANONICALIZATION_VERSION,
        "source_identity_schema_version": frv.FRV_SOURCE_IDENTITY_SCHEMA_VERSION,
        "canonical_point_content_spec_version": frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
        "scope_schema_version": frv.FRV_SCOPE_DEFINITION_SCHEMA_VERSION,
        "scope_construction_version": "scope-1",
        "scope_geometry_encoding_version": frv.FRV_SCOPE_GEOMETRY_ENCODING_SCHEMA_VERSION,
        "coverage_manifest_schema_version": frv.FRV_SCOPE_COVERAGE_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_version": frv.FRV_MEMBERSHIP_ARTIFACT_SCHEMA_VERSION,
        "block_schema_version": frv.FRV_ARTIFACT_BLOCK_SCHEMA_VERSION,
        "payload_schema_version": frv.FRV_ARTIFACT_PAYLOAD_SCHEMA_VERSION,
        "override_schema_version": frv.FRV_MANUAL_OVERRIDE_SCHEMA_VERSION,
        "scope_mode": frv.FrvScopeMode.EXHAUSTIVE_REGION,
        "scope_policy_digest": SHA,
        "override_policy_digest": SHA,
        "feature_policy_digest": SHA,
        "resource_policy_digest": SHA,
        "hag_provider_interface_version": "hag-v1",
        "geometry_provider_interface_version": "geom-v1",
        "scan_provider_interface_version": "scan-v1",
        "hag_provider_config_digest": SHA,
        "geometry_provider_config_digest": SHA,
        "scan_provider_config_digest": SHA,
        "wire_provider_interface_version": None,
        "wire_provider_config_digest": None,
        "feature_policy_version": "fp-1",
        "membership_parameter_schema_version": "mps-1",
        "calibration_record_digest": None,
        "allow_wire_body_hard_exclusion": False,
        "allow_wire_body_confirmed": False,
        "allow_review_routing_reduction": False,
        "allow_source_specific_selector": False,
        "configuration_purpose": frv.FrvConfigurationPurpose.TEST_ONLY,
        "configuration_is_runtime_complete": False,
        "formal_completeness_allowed": False,
        "membership_artifact_publication_allowed": False,
    }
    payload["configuration_digest"] = frv.frv_digest(payload)
    return frv.FrvConfiguration.from_dict(payload)


def _reference(
    path: Path,
    identity: frv.SourceIdentity,
    config: FrvSourceProviderConfig,
    configuration: frv.FrvConfiguration,
    *,
    component_sha: str,
    component_size: int,
    canonical_spec=None,
) -> FrvSourceProviderReference:
    component = FrvA2ArtifactComponentBinding(
        component_key="source_file",
        component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
        sha256=component_sha,
        size_bytes=component_size,
    )
    artifact_id = (
        f"frv-a2-ref-v1|source_scan|source_scan|"
        f"source_file,source_file,{component_sha},{component_size}"
    )
    return FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=artifact_id,
        artifact_components=(component,),
        runtime_access_path=path,
        source_identity=identity,
        canonical_point_content_spec=canonical_spec,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest=config.provider_config_digest,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )


def _expected_frvpcd1(path: Path, count: int, spec) -> str:
    with laspy.open(path, mode="r") as reader:
        pts = reader.read()
    xyz = np.stack([np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)], axis=1)
    chunk = _chunk_with_points(xyz.tolist(), list(range(count)))
    return derive_frv_source_point_content_digest([chunk], canonical_point_content_spec=spec)


def _setup(
    path: Path,
    *,
    count: int = 7,
    pass_through_dimensions=(),
    canonical_spec=None,
) -> tuple[FrvSourceProviderReference, FrvSourceProviderConfig, frv.FrvConfiguration]:
    write_las(path, point_count=count)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    canonical_digest = None
    canonical_spec_digest = None
    canonical_spec_version = None
    if canonical_spec is not None:
        canonical_digest = _expected_frvpcd1(path, count, canonical_spec)
        canonical_spec_digest = canonical_spec.spec_digest
        canonical_spec_version = frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION
    config = _provider_config(pass_through_dimensions=pass_through_dimensions)
    configuration = _configuration()
    identity = _source_identity(
        str(path),
        file_sha,
        count,
        _bounds_for(count),
        canonical_digest=canonical_digest,
        canonical_spec_digest=canonical_spec_digest,
        canonical_spec_version=canonical_spec_version,
    )
    reference = _reference(
        path,
        identity,
        config,
        configuration,
        component_sha=file_sha,
        component_size=file_size,
        canonical_spec=canonical_spec,
    )
    return reference, config, configuration


# ---------------------------------------------------------------------------
# API and dependency
# ---------------------------------------------------------------------------


def test_laspy_version_is_2_7_0() -> None:
    assert laspy.__version__ == "2.7.0"


def test_load_frv_source_public_export() -> None:
    assert callable(load_frv_source)
    assert hasattr(source_provider, "_FrvSourceLoadSession")


def test_private_session_not_exported() -> None:
    from algorithms import io

    assert not hasattr(io, "_FrvSourceLoadSession")


def test_load_frv_source_signature() -> None:
    import inspect

    parameters = list(inspect.signature(load_frv_source).parameters)
    assert parameters == ["reference", "frv_configuration", "provider_config", "chunk_size"]
    signature = inspect.signature(load_frv_source)
    assert signature.parameters["frv_configuration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["provider_config"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["chunk_size"].default == 1000000


def test_load_frv_source_rejects_bad_types(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    with pytest.raises(TypeError):
        load_frv_source("not-a-reference", frv_configuration=configuration, provider_config=config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        load_frv_source(reference, frv_configuration="x", provider_config=config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        load_frv_source(reference, frv_configuration=configuration, provider_config="x")  # type: ignore[arg-type]


def test_load_frv_source_rejects_bad_chunk_size(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    for bad in (True, 0, -1, 1.5, "2"):
        with pytest.raises((TypeError, ValueError)):
            load_frv_source(
                reference,
                frv_configuration=configuration,
                provider_config=config,
                chunk_size=bad,  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Normal flow
# ---------------------------------------------------------------------------


def test_normal_flow_multi_chunk_and_last_short_chunk(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        chunks = list(session.iter_chunks())
        assert [c.point_count for c in chunks] == [3, 3, 1]
        assert [c.is_last for c in chunks] == [False, False, True]
        assert [c.start_index for c in chunks] == [0, 3, 6]
        result = session.finalize()
    contracts.validate_finalized_frv_source_provider_result(result)
    evidence = contracts.to_frv_a2_provider_binding_evidence(result)
    contracts.validate_frv_a2_provider_binding_evidence(evidence)
    assert result.proof.source_point_count == 7


def test_single_point_file(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=1)
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        chunks = list(session.iter_chunks())
        assert len(chunks) == 1
        assert chunks[0].point_count == 1
        assert chunks[0].is_last is True
        assert chunks[0].original_index.tolist() == [0]
        result = session.finalize()
    assert result.proof.source_point_count == 1


def test_chunk_contract(tmp_path: Path) -> None:
    reference, config, configuration = _setup(
        tmp_path / "s.las",
        count=7,
        pass_through_dimensions=(
            FrvSourcePassThroughDimension.CLASSIFICATION,
            FrvSourcePassThroughDimension.INTENSITY,
        ),
    )
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        chunks = list(session.iter_chunks())
        all_index = np.concatenate([c.original_index for c in chunks])
        assert all_index.tolist() == list(range(7))
        for chunk in chunks:
            assert chunk.xyz.dtype == np.dtype("<f8")
            assert chunk.xyz.shape == (chunk.point_count, 3)
            assert chunk.xyz.flags["C_CONTIGUOUS"]
            assert not chunk.xyz.flags.writeable
            assert bool(np.isfinite(chunk.xyz).all())
            assert chunk.original_index.dtype == np.dtype("<i8")
            assert not chunk.original_index.flags.writeable
            assert list(chunk.attributes.keys()) == ["classification", "intensity"]
            assert chunk.attributes["classification"].dtype == np.dtype("|u1")
            assert chunk.attributes["intensity"].dtype == np.dtype("<u2")
            assert not chunk.attributes["classification"].flags.writeable
            assert not chunk.attributes["intensity"].flags.writeable
        result = session.finalize()
    assert result.proof.header.point_count == 7


def test_header_metadata_matches_las(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    header = result.proof.header
    assert header.point_count == 7
    assert header.source_format.value == "las"
    assert header.min_xyz == (100.0, 200.0, 0.0)
    assert header.max_xyz == (106.0, 206.0, 6.0)
    assert header.compression_state.value == "uncompressed"


def test_close_idempotent(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    session.close()
    session.close()


def test_context_manager_and_result_survives_close(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    with _load(reference, config, configuration, chunk_size=3) as session:
        list(session.iter_chunks())
        result = session.finalize()
    assert result.proof.source_point_count == 7
    contracts.validate_finalized_frv_source_provider_result(result)
    evidence = contracts.to_frv_a2_provider_binding_evidence(result)
    assert evidence.domain_count == 7


def test_canonical_content_digest_matches_authority(tmp_path: Path) -> None:
    spec = _exact_canonical_spec()
    reference, config, configuration = _setup(tmp_path / "s.las", count=5, canonical_spec=spec)
    session = _load(reference, config, configuration, chunk_size=2)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    assert result.proof.canonical_point_content_digest == _expected_frvpcd1(
        tmp_path / "s.las", 5, spec
    )
    assert result.proof.canonical_point_content_spec_digest == spec.spec_digest


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_iter_chunks_repeat_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    list(session.iter_chunks())
    with pytest.raises(FrvSourceLoadError):
        list(session.iter_chunks())
    session.close()


def test_finalize_before_iteration_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    with pytest.raises(FrvSourceLoadError) as excinfo:
        session.finalize()
    assert "session_state" in str(excinfo.value)
    session.close()


def test_finalize_after_partial_iteration_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=2)
    iterator = session.iter_chunks()
    next(iterator)
    with pytest.raises(FrvSourceLoadError):
        session.finalize()
    session.close()
    iterator.close()


def test_finalize_success_then_repeat_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
        with pytest.raises(FrvSourceLoadError):
            session.finalize()
    assert result.proof.source_point_count == 7


def test_methods_fail_after_close(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    session.close()
    with pytest.raises(FrvSourceLoadError):
        list(session.iter_chunks())
    with pytest.raises(FrvSourceLoadError):
        session.finalize()


def test_exit_returns_false_and_does_not_swallow(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    with pytest.raises(RuntimeError):
        with session:
            raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Format and data errors
# ---------------------------------------------------------------------------


def test_file_missing_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing.las"
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), SHA, 5, _bounds_for(5))
    reference = _reference(
        path, identity, config, configuration, component_sha=SHA, component_size=123
    )
    with pytest.raises(FrvSourceLoadError) as excinfo:
        load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "runtime_access_path" in str(excinfo.value)
    assert "FRV_PROVIDER_MISSING" in str(excinfo.value)


def test_zero_point_source_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    write_las(path, point_count=0)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), file_sha, 0, _bounds_for(0))
    reference = _reference(
        path, identity, config, configuration, component_sha=file_sha, component_size=file_size
    )
    with pytest.raises(FrvSourceLoadError) as excinfo:
        load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "source_point_count" in str(excinfo.value)


def test_missing_required_dimension_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    write_las(path, point_count=5, point_format=0)  # point format 0 has no gps_time
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config(pass_through_dimensions=(FrvSourcePassThroughDimension.GPS_TIME,))
    configuration = _configuration()
    identity = _source_identity(str(path), file_sha, 5, _bounds_for(5))
    reference = _reference(
        path, identity, config, configuration, component_sha=file_sha, component_size=file_size
    )
    session = _load(reference, config, configuration, chunk_size=2)
    with pytest.raises(FrvSourceLoadError) as excinfo:
        list(session.iter_chunks())
    assert "source_format" in str(excinfo.value)
    session.close()


def test_duplicate_pass_through_declaration_rejected() -> None:
    with pytest.raises(ValueError):
        _provider_config(
            pass_through_dimensions=(
                FrvSourcePassThroughDimension.CLASSIFICATION,
                FrvSourcePassThroughDimension.CLASSIFICATION,
            )
        )


def test_nonfinite_coordinate_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=5)
    session = _load(reference, config, configuration, chunk_size=2)

    class _NanRaw:
        x = np.array([float("nan")], dtype=np.float64)
        y = np.array([0.0], dtype=np.float64)
        z = np.array([0.0], dtype=np.float64)
        array = np.zeros(1)

    session._reader.chunk_iterator = lambda chunk_size: iter([_NanRaw()])
    with pytest.raises(FrvSourceLoadError) as excinfo:
        list(session.iter_chunks())
    assert "xyz" in str(excinfo.value)
    session.close()


def test_point_count_mismatch_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=2)
    real_iter = session._reader.chunk_iterator

    def dropped_first_iter(chunk_size):
        first = True
        for chunk in real_iter(chunk_size):
            if first:
                first = False
                continue
            yield chunk

    session._reader.chunk_iterator = dropped_first_iter
    with pytest.raises(FrvSourceLoadError) as excinfo:
        list(session.iter_chunks())
    assert "FRV_STREAM_INCOMPLETE" in str(excinfo.value)
    session.close()


def test_chunk_read_failure_aborts(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=2)

    def failing_iter(chunk_size):
        raise OSError("simulated read failure")

    session._reader.chunk_iterator = failing_iter
    with pytest.raises(FrvSourceLoadError) as excinfo:
        list(session.iter_chunks())
    assert "FRV_STREAM_INCOMPLETE" in str(excinfo.value)
    # after the iterator failure, finalize must also fail
    with pytest.raises(FrvSourceLoadError):
        session.finalize()
    session.close()


def test_compressed_format_rejected(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=5)
    with mock.patch(
        "algorithms.io.source_provider.laspy.open",
        return_value=_FakeReader(_CompressedHeader()),
    ):
        with pytest.raises(FrvSourceLoadError) as excinfo:
            load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "source_format" in str(excinfo.value)


class _CompressedHeader:
    are_points_compressed = True
    evlrs = []
    vlrs = []
    point_count = 5
    version = "1.4"
    scales = [0.01, 0.01, 0.01]
    offsets = [100.0, 200.0, 0.0]
    mins = [100.0, 200.0, 0.0]
    maxs = [104.0, 204.0, 4.0]

    def parse_crs(self):
        return None


class _FakeReader:
    def __init__(self, header):
        self.header = header

    def close(self):
        pass


# ---------------------------------------------------------------------------
# TOCTOU and resources
# ---------------------------------------------------------------------------


def test_toctou_hash_changed(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    reference, config, configuration = _setup(path, count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        list(session.iter_chunks())
        path.write_bytes(b"different-content-" * 100)
        with pytest.raises(FrvSourceLoadError) as excinfo:
            session.finalize()
        assert "FRV_ARTIFACT_DIGEST_MISMATCH" in str(excinfo.value)


def test_toctou_size_changed(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    reference, config, configuration = _setup(path, count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        list(session.iter_chunks())
        path.write_bytes(b"tiny")
        with pytest.raises(FrvSourceLoadError) as excinfo:
            session.finalize()
        assert "FRV_ARTIFACT_DIGEST_MISMATCH" in str(excinfo.value)


def test_reference_digest_mismatch_rejected_before_open(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    write_las(path, point_count=5)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), file_sha, 5, _bounds_for(5))
    # component sha differs from the identity -> reference validator rejects
    reference = _reference(
        path, identity, config, configuration, component_sha="b" * 64, component_size=file_size
    )
    with mock.patch("builtins.open", side_effect=AssertionError("open called before validation")):
        with pytest.raises(FrvSourceLoadError) as excinfo:
            load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "FRV_PROVIDER_INVALID" in str(excinfo.value)


def test_reader_closed_before_handle(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    order: list[str] = []
    reader_close = session._reader.close
    handle_close = session._handle.close

    def rec_reader():
        order.append("reader")
        reader_close()

    def rec_handle():
        order.append("handle")
        handle_close()

    session._reader.close = rec_reader
    session._handle.close = rec_handle
    session.close()
    assert order == ["reader", "handle"]


def test_first_failure_wins_over_cleanup(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=2)

    def failing_iter(chunk_size):
        raise OSError("read failure")

    session._reader.chunk_iterator = failing_iter

    def failing_reader_close():
        raise OSError("cleanup failure")

    session._reader.close = failing_reader_close
    with pytest.raises(FrvSourceLoadError) as excinfo:
        list(session.iter_chunks())
    # the stream failure wins; the cleanup failure is swallowed
    assert "FRV_STREAM_INCOMPLETE" in str(excinfo.value)


def test_close_failure_after_finalize(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las")
    session = _load(reference, config, configuration, chunk_size=3)
    list(session.iter_chunks())

    def failing_close():
        raise OSError("close failure")

    session._handle.close = failing_close
    with pytest.raises(FrvSourceLoadError) as excinfo:
        session.finalize()
    assert "FRV_PROVIDER_INVALID" in str(excinfo.value)


def test_raw_exception_and_path_not_leaked(tmp_path: Path) -> None:
    path = tmp_path / "missing.las"
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), SHA, 5, _bounds_for(5))
    reference = _reference(
        path, identity, config, configuration, component_sha=SHA, component_size=123
    )
    with pytest.raises(FrvSourceLoadError) as excinfo:
        load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "FileNotFoundError" not in str(excinfo.value)
    assert "missing.las" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


# ---------------------------------------------------------------------------
# Streaming and determinism
# ---------------------------------------------------------------------------


def test_no_laspy_read_and_single_consumption(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=2)
    with mock.patch.object(
        source_provider.laspy, "read", side_effect=AssertionError("laspy.read forbidden")
    ):
        with session:
            list(session.iter_chunks())
            session.finalize()


def test_chunk_size_invariance(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    reference, config, configuration = _setup(path, count=7, canonical_spec=_exact_canonical_spec())
    results = []
    for chunk_size in (2, 5):
        session = _load(reference, config, configuration, chunk_size=chunk_size)
        with session:
            list(session.iter_chunks())
            results.append(session.finalize())
    assert results[0].proof == results[1].proof
    assert results[0] == results[1]


def test_repeated_run_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    reference, config, configuration = _setup(path, count=7)
    digests = []
    for _ in range(2):
        session = _load(reference, config, configuration, chunk_size=3)
        with session:
            list(session.iter_chunks())
            result = session.finalize()
        digests.append(
            (
                result.proof.source_content_identity_digest,
                result.proof.original_index_domain_digest,
                result.proof.canonical_point_content_digest,
            )
        )
    assert digests[0] == digests[1]


def test_finalized_result_and_bnd1_valid(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    contracts.validate_finalized_frv_source_provider_result(result)
    evidence = contracts.to_frv_a2_provider_binding_evidence(result)
    contracts.validate_frv_a2_provider_binding_evidence(evidence)
    assert evidence.provider_role is FrvA2ProviderRole.SOURCE_SCAN
    assert evidence.provider_readiness.value == "validated"
    assert evidence.binding_digest == contracts.frv_a2_provider_binding_digest(evidence)


def write_las_with_vlrs(path: Path, *, point_count: int = 5, vlrs=()) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = np.array([0.01, 0.01, 0.01])
    header.offsets = np.array([100.0, 200.0, 0.0])
    for user_id, record_id, data in vlrs:
        header.vlrs.append(
            laspy.VLR(
                user_id=user_id,
                record_id=record_id,
                description="",
                record_data=data,
            )
        )
    las = laspy.LasData(header)
    idx = np.arange(point_count)
    las.x = 100.0 + idx.astype(np.float64)
    las.y = 200.0 + idx.astype(np.float64)
    las.z = idx.astype(np.float64)
    las.write(path)


# ---------------------------------------------------------------------------
# Final-review findings coverage (BLK-001/002, MAJ-003..008)
# ---------------------------------------------------------------------------


def test_reader_closed_before_final_hash_and_same_handle(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    original_handle = session._handle
    assert original_handle is not None
    list(session.iter_chunks())
    assert session._reader is None  # reader closed at natural stream completion
    assert session._handle is original_handle  # same handle used for the final hash
    session.finalize()
    assert session._handle is None  # handle closed by finalize
    session.close()  # idempotent after finalize


def test_normal_iteration_without_finalize_close_aborts(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    list(session.iter_chunks())  # STREAM_COMPLETE, not finalized
    session.close()
    with pytest.raises(FrvSourceLoadError):
        session.finalize()


def test_active_exception_not_overridden_by_cleanup(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)

    def failing_close():
        raise OSError("cleanup failure")

    session._handle.close = failing_close
    with pytest.raises(RuntimeError, match="boom"):
        with session:
            raise RuntimeError("boom")


def test_generator_created_but_not_started_second_iter_chunks_rejected(
    tmp_path: Path,
) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    generator = session.iter_chunks()  # single-use locked at call time
    with pytest.raises(FrvSourceLoadError):
        session.iter_chunks()  # second call rejected
    session.close()
    generator.close()


def test_laz_suffix_rejected_before_open(tmp_path: Path) -> None:
    path = tmp_path / "source.laz"
    write_las(path, point_count=5)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), file_sha, 5, _bounds_for(5))
    reference = _reference(
        path, identity, config, configuration, component_sha=file_sha, component_size=file_size
    )
    with mock.patch("builtins.open", side_effect=AssertionError("open called")):
        with pytest.raises(FrvSourceLoadError) as excinfo:
            load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "source_format" in str(excinfo.value)


def test_final_hash_io_failure_reference_invalid(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    list(session.iter_chunks())

    def failing_seek(*args):
        raise OSError("seek failure")

    session._handle.seek = failing_seek
    with pytest.raises(FrvSourceLoadError) as excinfo:
        session.finalize()
    assert "FRV_PROVIDER_INVALID" in str(excinfo.value)
    session.close()


def test_stream_index_domain_digest_matches_authority() -> None:
    from algorithms.io.source_provider import _stream_index_domain_digest

    for n in (1, 7, 100):
        assert _stream_index_domain_digest(n, SHA) == contracts.frv_a2_index_domain_digest(
            "ORIGINAL_INDEX",
            SHA,
            contracts.FrvA2DomainKind.FULL_SOURCE,
            tuple(range(n)),
        )


def test_index_digest_not_materialized(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=2)
    with mock.patch.object(
        source_provider,
        "_stream_index_domain_digest",
        wraps=source_provider._stream_index_domain_digest,
    ) as mocked:
        with session:
            list(session.iter_chunks())
            result = session.finalize()
        assert mocked.call_count == 1
        assert isinstance(mocked.call_args.args[0], int)  # total, not a full N tuple
    assert result.proof.original_index_domain_digest == contracts.frv_a2_index_domain_digest(
        "ORIGINAL_INDEX",
        reference.source_identity.source_content_identity_digest,
        contracts.FrvA2DomainKind.FULL_SOURCE,
        tuple(range(7)),
    )


def test_non_crs_vlr_count(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "plain.las", count=5)
    session = _load(reference, config, configuration, chunk_size=2)
    with session:
        list(session.iter_chunks())
        result = session.finalize()
    assert result.proof.header.non_crs_vlr_count == 0

    path = tmp_path / "vlrs.las"
    write_las_with_vlrs(
        path,
        point_count=5,
        vlrs=[("custom", 100, b"a"), ("other", 101, b"bb")],
    )
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config2 = _provider_config()
    configuration2 = _configuration()
    identity = _source_identity(str(path), file_sha, 5, _bounds_for(5))
    reference2 = _reference(
        path, identity, config2, configuration2, component_sha=file_sha, component_size=file_size
    )
    session2 = _load(reference2, config2, configuration2, chunk_size=2)
    with session2:
        list(session2.iter_chunks())
        result2 = session2.finalize()
    assert result2.proof.header.non_crs_vlr_count == 2


def test_crs_vlr_rejected(tmp_path: Path) -> None:
    path = tmp_path / "crs.las"
    write_las_with_vlrs(path, point_count=5, vlrs=[("LASF_Projection", 9999, b"\x00" * 8)])
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), file_sha, 5, _bounds_for(5))
    reference = _reference(
        path, identity, config, configuration, component_sha=file_sha, component_size=file_size
    )
    with pytest.raises(FrvSourceLoadError) as excinfo:
        load_frv_source(reference, frv_configuration=configuration, provider_config=config)
    assert "FRV_SOURCE_IDENTITY_FAILED" in str(excinfo.value)  # COORDINATE_AUTHORITY_REJECTED


def test_forged_reference_provider_digest_rejected_before_open(tmp_path: Path) -> None:
    path = tmp_path / "s.las"
    write_las(path, point_count=5)
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    file_size = path.stat().st_size
    config = _provider_config()
    configuration = _configuration()
    identity = _source_identity(str(path), file_sha, 5, _bounds_for(5))
    reference = _reference(
        path, identity, config, configuration, component_sha=file_sha, component_size=file_size
    )
    forged = FrvSourceProviderReference(
        provider_role=FrvA2ProviderRole.SOURCE_SCAN,
        provider_key="source_scan",
        artifact_reference_id=reference.artifact_reference_id,
        artifact_components=reference.artifact_components,
        runtime_access_path=path,
        source_identity=identity,
        canonical_point_content_spec=None,
        configuration_digest=configuration.configuration_digest,
        provider_config_digest="b" * 64,
        interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
        schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
        algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    )
    with mock.patch("builtins.open", side_effect=AssertionError("open called")):
        with pytest.raises(FrvSourceLoadError) as excinfo:
            load_frv_source(forged, frv_configuration=configuration, provider_config=config)
    assert "FRV_PROVIDER_INVALID" in str(excinfo.value)


def test_no_result_on_failure_paths(tmp_path: Path) -> None:
    reference, config, configuration = _setup(tmp_path / "s.las", count=7)
    session = _load(reference, config, configuration, chunk_size=3)
    with pytest.raises(FrvSourceLoadError):
        session.finalize()  # never iterated -> SESSION_ABORTED
    assert session._result is None  # no partial Result is produced
    session.close()
