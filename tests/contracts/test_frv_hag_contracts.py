"""Focused FRV-A2 P-HAG contract tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_hag as hag
import numpy as np
import pytest
from core.contracts.frv_a2 import (
    FrvA2ArtifactComponentBinding,
    FrvA2ArtifactComponentKind,
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
)
from core.contracts.frv_hag import (
    FRV_HAG_GRID_DIGEST_DOMAIN,
    FRV_HAG_PROVIDER_KEY,
    FrvHagChunk,
    FrvHagLoadError,
    FrvHagProviderConfig,
    FrvHagProviderProof,
    FrvHagProviderResult,
    derive_frv_hag_provider_config_digest,
    frv_hag_ground_cell_inventory_digest,
    frv_hag_point_content_digest,
    validate_frv_hag_chunk,
    validate_frv_hag_provider_config,
)
from core.contracts.frv_source import FrvSourcePassThroughDimension

SHA = "a" * 64


def _config(**overrides) -> FrvHagProviderConfig:
    defaults: dict = {
        "schema_version": hag.FRV_HAG_CONFIG_SCHEMA_VERSION,
        "provider_key": FRV_HAG_PROVIDER_KEY,
        "provider_role": FrvA2ProviderRole.HAG,
        "interface_version": hag.FRV_HAG_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": hag.FRV_HAG_ALGORITHM_VERSION,
        "cell_size_m": 1.0,
        "support_radius_m": 5.0,
        "min_support_cells": 1,
        "max_support_cells": 8,
        "negative_hag_tolerance_m": 0.25,
        "classification_dimension": "classification",
    }
    explicit = overrides.pop("provider_config_digest", None)
    defaults.update(overrides)
    if explicit is not None:
        defaults["provider_config_digest"] = explicit
    else:
        defaults["provider_config_digest"] = derive_frv_hag_provider_config_digest(
            schema_version=defaults["schema_version"],
            provider_key=defaults["provider_key"],
            provider_role=defaults["provider_role"],
            interface_version=defaults["interface_version"],
            algorithm_version=defaults["algorithm_version"],
            cell_size_m=defaults["cell_size_m"],
            support_radius_m=defaults["support_radius_m"],
            min_support_cells=defaults["min_support_cells"],
            max_support_cells=defaults["max_support_cells"],
            negative_hag_tolerance_m=defaults["negative_hag_tolerance_m"],
            classification_dimension=defaults["classification_dimension"],
        )
    return FrvHagProviderConfig(**defaults)


def _chunk(**overrides) -> FrvHagChunk:
    point_count = overrides.get("point_count", 3)
    defaults: dict = {
        "source_id": "scene-1",
        "chunk_id": 0,
        "start_index": 0,
        "point_count": point_count,
        "original_index": _ro(np.arange(point_count, dtype="<i8")),
        "ground_support": _ro(np.ones(point_count, dtype=bool)),
        "support_cell_count": _ro(np.ones(point_count, dtype="<u2")),
        "ground_z_m": _ro(np.zeros(point_count, dtype="<f8")),
        "raw_hag_m": _ro(np.ones(point_count, dtype="<f8")),
        "clamped_hag_m": _ro(np.ones(point_count, dtype="<f8")),
        "hag_evidence_state": _ro(
            np.full(point_count, int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE), dtype="|u1")
        ),
        "reason_flags": _ro(np.zeros(point_count, dtype="<u4")),
        "quality_flags": _ro(np.zeros(point_count, dtype="<u4")),
        "is_last": True,
    }
    defaults.update(overrides)
    return FrvHagChunk(**defaults)


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _state(value) -> np.ndarray:
    """Single-row read-only ``|u1`` evidence-state array."""
    return _ro(np.array([int(value)], dtype="|u1"))


def test_config_defaults_and_exact_digest() -> None:
    config = _config()
    assert config.schema_version == hag.FRV_HAG_CONFIG_SCHEMA_VERSION
    assert config.provider_key == "hag_provider"
    assert config.provider_role is FrvA2ProviderRole.HAG
    assert config.cell_size_m == 1.0
    assert config.support_radius_m == 5.0
    assert config.min_support_cells == 1
    assert config.max_support_cells == 8
    assert config.negative_hag_tolerance_m == 0.25
    assert config.classification_dimension == "classification"
    validate_frv_hag_provider_config(config)
    assert config.provider_config_digest == derive_frv_hag_provider_config_digest(
        schema_version=config.schema_version,
        provider_key=config.provider_key,
        provider_role=config.provider_role,
        interface_version=config.interface_version,
        algorithm_version=config.algorithm_version,
        cell_size_m=config.cell_size_m,
        support_radius_m=config.support_radius_m,
        min_support_cells=config.min_support_cells,
        max_support_cells=config.max_support_cells,
        negative_hag_tolerance_m=config.negative_hag_tolerance_m,
        classification_dimension=config.classification_dimension,
    )


def test_config_exact_field_order() -> None:
    assert [f.name for f in dataclasses.fields(FrvHagProviderConfig)] == [
        "schema_version",
        "provider_key",
        "provider_role",
        "interface_version",
        "algorithm_version",
        "cell_size_m",
        "support_radius_m",
        "min_support_cells",
        "max_support_cells",
        "negative_hag_tolerance_m",
        "classification_dimension",
        "provider_config_digest",
    ]


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        _config(cell_size_m=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(cell_size_m=1)  # type: ignore[arg-type]  # int not float
    with pytest.raises(ValueError):
        _config(min_support_cells=0)
    with pytest.raises(ValueError):
        _config(min_support_cells=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(min_support_cells=np.int64(2))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        _config(support_radius_m=0.5)  # below cell_size
    with pytest.raises(ValueError):
        _config(max_support_cells=0)  # below min_support_cells
    with pytest.raises(ValueError):
        _config(classification_dimension="other")
    with pytest.raises(ValueError):
        _config(provider_role=FrvA2ProviderRole.SOURCE_SCAN)
    with pytest.raises(ValueError):
        _config(provider_config_digest="not-a-sha")


def test_config_frozen() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.cell_size_m = 2.0  # type: ignore[misc]


def test_config_from_dict_exact_keys() -> None:
    config = _config()
    restored = FrvHagProviderConfig.from_dict(config.to_dict())
    assert restored == config
    payload = config.to_dict()
    del payload["provider_key"]
    with pytest.raises(ValueError, match="missing"):
        FrvHagProviderConfig.from_dict(payload)
    payload = config.to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvHagProviderConfig.from_dict(payload)


def test_chunk_normal_construction_and_read_only() -> None:
    chunk = _chunk()
    assert chunk.point_count == 3
    assert chunk.is_last is True
    assert not chunk.original_index.flags.writeable
    assert chunk.original_index.dtype == np.dtype("<i8")
    assert chunk.ground_z_m.dtype == np.dtype("<f8")
    assert chunk.hag_evidence_state.dtype == np.dtype("|u1")
    assert chunk.reason_flags.dtype == np.dtype("<u4")
    assert chunk.quality_flags.dtype == np.dtype("<u4")
    with pytest.raises(FrozenInstanceError):
        chunk.source_id = "other"  # type: ignore[misc]


def test_chunk_exact_field_order() -> None:
    assert [f.name for f in dataclasses.fields(FrvHagChunk)] == [
        "source_id",
        "chunk_id",
        "start_index",
        "point_count",
        "original_index",
        "ground_support",
        "support_cell_count",
        "ground_z_m",
        "raw_hag_m",
        "clamped_hag_m",
        "hag_evidence_state",
        "reason_flags",
        "quality_flags",
        "is_last",
    ]


def test_chunk_rejects_empty_and_bad_arrays() -> None:
    with pytest.raises(ValueError):
        _chunk(point_count=0)
    with pytest.raises(ValueError):
        _chunk(point_count=-1)
    with pytest.raises(ValueError):
        _chunk(ground_z_m=_ro(np.array([float("nan"), 0.0, 0.0], dtype="<f8")))
    with pytest.raises(ValueError):
        _chunk(ground_z_m=_ro(np.array([float("inf"), 0.0, 0.0], dtype="<f8")))
    with pytest.raises(ValueError):
        _chunk(ground_z_m=np.zeros(3, dtype="<f8"))  # writable
    # non-increasing original_index is rejected by the chunk validator
    non_increasing = _chunk(original_index=_ro(np.zeros(3, dtype="<i8")))
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_frv_hag_chunk(non_increasing, provider_config=_config())


def test_chunk_rejects_negative_zero() -> None:
    negative_zero = np.zeros(3, dtype="<f8")
    negative_zero[0] = -0.0
    negative_zero.setflags(write=False)
    with pytest.raises(ValueError, match="negative zero"):
        _chunk(raw_hag_m=negative_zero)
    with pytest.raises(ValueError, match="negative zero"):
        _chunk(ground_z_m=negative_zero)
    with pytest.raises(ValueError, match="negative zero"):
        _chunk(clamped_hag_m=negative_zero)


def test_chunk_validator_unsupported_and_supported() -> None:
    config = _config()
    unsupported = _chunk(
        ground_support=_ro(np.zeros(1, dtype=bool)),
        support_cell_count=_ro(np.zeros(1, dtype="<u2")),
        ground_z_m=_ro(np.zeros(1, dtype="<f8")),
        raw_hag_m=_ro(np.zeros(1, dtype="<f8")),
        clamped_hag_m=_ro(np.zeros(1, dtype="<f8")),
        hag_evidence_state=_ro(
            np.array([int(frv.FrvHagEvidenceState.HAG_MISSING)], dtype="|u1")
        ),
        reason_flags=_ro(
            np.array(
                [int(frv.FrvReasonFlag.INSUFFICIENT_EVIDENCE | frv.FrvReasonFlag.HAG_MISSING)],
                dtype="<u4",
            )
        ),
        point_count=1,
        original_index=_ro(np.zeros(1, dtype="<i8")),
        is_last=True,
    )
    validate_frv_hag_chunk(unsupported, provider_config=config)
    supported = _chunk()
    validate_frv_hag_chunk(supported, provider_config=config)
    bad = _chunk(
        ground_support=_ro(np.array([False, True, True], dtype=bool)),
        support_cell_count=_ro(np.array([0, 0, 1], dtype="<u2")),
    )
    with pytest.raises(ValueError):
        validate_frv_hag_chunk(bad, provider_config=config)


def test_ground_inventory_digest_stable() -> None:
    d1 = frv_hag_ground_cell_inventory_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        ground_point_count=4,
        cells=((2, 3, 0.0, 1), (1, 1, -0.25, 2), (0, 0, 0.0, 1)),
    )
    d2 = frv_hag_ground_cell_inventory_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        ground_point_count=4,
        cells=((0, 0, 0.0, 1), (1, 1, -0.25, 2), (2, 3, 0.0, 1)),
    )
    assert d1 == d2  # sorted independent of input order
    assert FRV_HAG_GRID_DIGEST_DOMAIN in "FRVHAGGRID1"


def test_point_content_digest_oracle() -> None:
    chunk = _chunk(point_count=2, is_last=True)
    digest = frv_hag_point_content_digest(
        source_content_identity_digest=SHA,
        source_identity_record_digest=SHA,
        provider_config_digest=SHA,
        source_point_count=2,
        chunks=[chunk],
    )
    assert len(digest) == 64
    with pytest.raises(ValueError):
        frv_hag_point_content_digest(
            source_content_identity_digest=SHA,
            source_identity_record_digest=SHA,
            provider_config_digest=SHA,
            source_point_count=3,  # chunks only cover 0..1
            chunks=[chunk],
        )


def test_result_no_public_constructor() -> None:
    with pytest.raises(TypeError):
        FrvHagProviderResult()  # type: ignore[call-arg]


def test_result_factory_and_capability() -> None:
    from tests.contracts.test_frv_source_contracts import _source_identity

    source_identity = _source_identity(path=r"C:\__hag__\s.las")
    proof = _proof_for_identity(source_identity)
    # build a proof that passes source-identity binding
    proof = _proof_for_identity(source_identity)
    result = hag._create_finalized_frv_hag_provider_result(
        proof, source_identity, capability=hag._FRV_HAG_FINALIZED_CAPABILITY
    )
    hag.validate_finalized_frv_hag_provider_result(result)
    assert "capability" not in repr(result)
    with pytest.raises(ValueError, match="capability"):
        hag._create_finalized_frv_hag_provider_result(
            proof, source_identity, capability="wrong"
        )
    forged = object.__new__(FrvHagProviderResult)
    object.__setattr__(forged, "proof", proof)
    object.__setattr__(forged, "source_identity", source_identity)
    object.__setattr__(forged, "_finalized_capability", object())
    with pytest.raises(ValueError, match="capability"):
        hag.validate_finalized_frv_hag_provider_result(forged)


def test_load_error_construction() -> None:
    record = FrvA2FailureRecord(
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        FrvA2ProviderRole.HAG,
        "hag_provider",
        "source_result",
        frv.FrvErrorCode.FRV_PROVIDER_INVALID,
        int(frv.FrvReasonFlag.PROVIDER_UNCERTAINTY),
        "structure_integrity:hag:hag_provider:source_result:FRV_PROVIDER_INVALID:1024",
    )
    error = FrvHagLoadError(
        failure_record=record,
        quality_flags=int(frv.FrvQualityFlag.PROVIDER_INVALID),
    )
    assert str(error) == record.detail
    assert error.args == (record.detail,)
    with pytest.raises(TypeError):
        FrvHagLoadError(failure_record="x", quality_flags=0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        FrvHagLoadError(failure_record=record, quality_flags=-1)


def _valid_proof() -> FrvHagProviderProof:
    from tests.contracts.test_frv_source_contracts import _source_identity

    return _proof_for_identity(_source_identity(path=r"C:\__hag__\s.las"))


def _proof_for_identity(source_identity) -> FrvHagProviderProof:
    source_content = source_identity.source_content_identity_digest
    source_record = source_identity.source_identity_record_digest
    count = int(source_identity.source_point_count)
    proof = FrvHagProviderProof.__new__(FrvHagProviderProof)
    fields = {
        "schema_version": hag.FRV_HAG_PROOF_SCHEMA_VERSION,
        "provider_key": "hag_provider",
        "provider_role": FrvA2ProviderRole.HAG,
        "interface_version": hag.FRV_HAG_PROVIDER_INTERFACE_VERSION,
        "algorithm_version": hag.FRV_HAG_ALGORITHM_VERSION,
        "configuration_digest": SHA,
        "provider_config_digest": _config().provider_config_digest,
        "source_artifact_reference_id": (
            f"frv-a2-ref-v1|source_scan|source_scan|source_file,source_file,{SHA},123"
        ),
        "artifact_reference_id": (
            f"frv-a2-ref-v1|hag|hag_provider|source_file,source_file,{SHA},123"
        ),
        "artifact_components": (
            FrvA2ArtifactComponentBinding(
                component_key="source_file",
                component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
                sha256=source_identity.source_file_sha256,
                size_bytes=123,
            ),
        ),
        "source_content_identity_digest": source_content,
        "source_identity_record_digest": source_record,
        "source_proof_digest": SHA,
        "source_binding_digest": SHA,
        "source_point_count": count,
        "domain_kind": FrvA2DomainKind.FULL_SOURCE,
        "domain_count": count,
        "original_index_domain_digest": contracts.frv_a2_index_domain_digest(
            "ORIGINAL_INDEX", source_content, FrvA2DomainKind.FULL_SOURCE, tuple(range(count))
        ),
        "hag_point_content_digest": SHA,
        "ground_cell_inventory_digest": SHA,
        "ground_point_count": 1,
        "supported_point_count": count,
        "unsupported_point_count": 0,
        "negative_tolerance_exceeded_point_count": 0,
        "hag_valid_positive_count": count,
        "hag_valid_near_zero_count": 0,
        "hag_within_negative_tolerance_count": 0,
        "hag_missing_count": 0,
        "hag_invalid_count": 0,
        "hag_out_of_domain_count": 0,
        "coordinate_frame": "LOCAL_CARTESIAN_XYZ",
        "units": "m",
        "origin_xyz": (0.0, 0.0, 0.0),
        "transform_declaration": "identity",
        "pass_through_dimensions": (FrvSourcePassThroughDimension.CLASSIFICATION,),
        "quality_flags": 0,
        "reason_flags": 0,
        "error_codes": (),
    }
    for name, value in fields.items():
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "proof_digest", proof.computed_digest())
    return proof


def _forged_proof(**tamper) -> FrvHagProviderProof:
    from tests.contracts.test_frv_source_contracts import _source_identity

    identity = _source_identity(path=r"C:\__hag__\s.las")
    proof = _proof_for_identity(identity)
    for name, value in tamper.items():
        object.__setattr__(proof, name, value)
    object.__setattr__(proof, "proof_digest", proof.computed_digest())
    return proof


def test_stream_index_domain_digest_matches_authority() -> None:
    from core.contracts.frv_hag import _stream_index_domain_digest

    for n in (1, 7, 100):
        assert _stream_index_domain_digest(n, SHA) == contracts.frv_a2_index_domain_digest(
            "ORIGINAL_INDEX", SHA, FrvA2DomainKind.FULL_SOURCE, tuple(range(n))
        )


def test_forged_rehashed_proof_rejected() -> None:
    from core.contracts.frv_hag import validate_frv_hag_provider_proof

    cases = [
        {"provider_role": FrvA2ProviderRole.SOURCE_SCAN},
        {"provider_key": "other"},
        {"schema_version": "wrong"},
        {"domain_kind": FrvA2DomainKind.DECLARED_SUBSET},
        {"coordinate_frame": "OTHER"},
        {"units": "ft"},
        {"transform_declaration": "non_identity"},
        {"source_point_count": 999},  # breaks conservation with counts
        {"hag_out_of_domain_count": 1},
        {"quality_flags": int(frv.FrvQualityFlag.PROVIDER_INVALID)},
        {"error_codes": (frv.FrvErrorCode.FRV_PROVIDER_INVALID,)},
        {  # forged component that no longer matches artifact_reference_id
            "artifact_components": (
                FrvA2ArtifactComponentBinding(
                    component_key="source_file",
                    component_kind=FrvA2ArtifactComponentKind.SOURCE_FILE,
                    sha256=SHA,
                    size_bytes=999,
                ),
            ),
        },
    ]
    for tamper in cases:
        forged = _forged_proof(**tamper)
        with pytest.raises(ValueError):
            validate_frv_hag_provider_proof(forged)


def test_stream_index_domain_digest_no_full_index_materialization(monkeypatch) -> None:
    """The P-HAG proof validator must never build ``tuple(range(N))``.

    ``_stream_index_domain_digest`` hashes the lazy ``range`` in-stream.
    The spy window is scoped to that pure hashing function (which calls
    neither ``tuple`` nor ``list``); builtins are restored in-body so pytest
    reporting always sees the real constructors.
    """
    import builtins

    from core.contracts.frv_hag import _stream_index_domain_digest

    real_tuple = builtins.tuple
    real_list = builtins.list
    large: list[int] = []

    def spy_tuple(arg):
        if isinstance(arg, range) and len(arg) > 1000:
            large.append(len(arg))
        return real_tuple(arg)

    def spy_list(arg):
        if isinstance(arg, range) and len(arg) > 1000:
            large.append(len(arg))
        return real_list(arg)

    monkeypatch.setattr(builtins, "tuple", spy_tuple)
    monkeypatch.setattr(builtins, "list", spy_list)
    try:
        digest = _stream_index_domain_digest(100_000, SHA)
    finally:
        monkeypatch.setattr(builtins, "tuple", real_tuple)
        monkeypatch.setattr(builtins, "list", real_list)
    assert len(digest) == 64
    assert large == []  # no tuple/list(range(100_000)) was ever constructed
    # spot-check the digest is stable against the authority framing
    assert _stream_index_domain_digest(100_000, SHA) == digest


def test_chunk_state_matrix() -> None:
    config = _config()
    missing = int(frv.FrvReasonFlag.INSUFFICIENT_EVIDENCE | frv.FrvReasonFlag.HAG_MISSING)
    invalid_quality = int(frv.FrvQualityFlag.PROVIDER_INVALID)
    invalid_reason = int(frv.FrvReasonFlag.HAG_INVALID)
    ood_reason = int(frv.FrvReasonFlag.HAG_OUT_OF_DOMAIN)

    def single(**overrides):
        return _chunk(
            point_count=1,
            original_index=_ro(np.zeros(1, dtype="<i8")),
            is_last=True,
            **overrides,
        )

    # valid state rows
    valid = [
        dict(hag_evidence_state=_state(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE)),
        dict(
            raw_hag_m=_ro(np.zeros(1, dtype="<f8")),
            clamped_hag_m=_ro(np.zeros(1, dtype="<f8")),
            hag_evidence_state=_state(frv.FrvHagEvidenceState.HAG_VALID_NEAR_ZERO),
        ),
        dict(
            raw_hag_m=_ro(np.array([-0.1], dtype="<f8")),
            clamped_hag_m=_ro(np.zeros(1, dtype="<f8")),
            hag_evidence_state=_state(frv.FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE),
        ),
        dict(
            ground_support=_ro(np.zeros(1, dtype=bool)),
            support_cell_count=_ro(np.zeros(1, dtype="<u2")),
            ground_z_m=_ro(np.zeros(1, dtype="<f8")),
            raw_hag_m=_ro(np.zeros(1, dtype="<f8")),
            clamped_hag_m=_ro(np.zeros(1, dtype="<f8")),
            hag_evidence_state=_state(frv.FrvHagEvidenceState.HAG_MISSING),
            reason_flags=_ro(np.array([missing], dtype="<u4")),
        ),
        dict(
            raw_hag_m=_ro(np.array([-0.5], dtype="<f8")),
            clamped_hag_m=_ro(np.zeros(1, dtype="<f8")),
            hag_evidence_state=_state(frv.FrvHagEvidenceState.HAG_INVALID),
            reason_flags=_ro(np.array([invalid_reason], dtype="<u4")),
            quality_flags=_ro(np.array([invalid_quality], dtype="<u4")),
        ),
        dict(
            ground_support=_ro(np.zeros(1, dtype=bool)),
            support_cell_count=_ro(np.zeros(1, dtype="<u2")),
            hag_evidence_state=_state(frv.FrvHagEvidenceState.HAG_OUT_OF_DOMAIN),
            reason_flags=_ro(np.array([ood_reason], dtype="<u4")),
            quality_flags=_ro(np.array([invalid_quality], dtype="<u4")),
        ),
    ]
    for overrides in valid:
        validate_frv_hag_chunk(single(**overrides), provider_config=config)

    # invalid state rows
    invalid = [
        # HAG_VALID_POSITIVE with raw=1 but support=False must be rejected
        dict(
            ground_support=_ro(np.zeros(1, dtype=bool)),
            hag_evidence_state=_ro(
                np.array([int(frv.FrvHagEvidenceState.HAG_VALID_POSITIVE)], dtype="|u1")
            ),
        ),
    ]
    for overrides in invalid:
        with pytest.raises(ValueError):
            validate_frv_hag_chunk(single(**overrides), provider_config=config)
