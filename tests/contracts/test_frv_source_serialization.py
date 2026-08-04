"""Focused FRV-A2 P-SOURCE C3-B proof serialization tests.

The proof is the only serializable C3-B/C3-C object; the finalized result is
explicitly non-serializable.  All fixtures are pure lexical contract data: no
file access, no laspy, no IO.
"""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import core.contracts as contracts
import core.contracts.frv as frv
import core.contracts.frv_source as source
import pytest

from tests.contracts.test_frv_source_contracts import (
    SHA,
    _exact_canonical_spec,
    _source_identity,
    _valid_proof,
    _valid_result,
)


def _payload() -> dict:
    return _valid_proof().to_dict()


def test_proof_to_dict_exact_keys() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    assert set(payload) == {f.name for f in dataclasses.fields(proof)}


def test_proof_to_dict_exact_key_order() -> None:
    proof = _valid_proof()
    assert list(proof.to_dict()) == [f.name for f in dataclasses.fields(proof)]


def test_proof_round_trip_and_deterministic() -> None:
    proof = _valid_proof()
    restored = source.FrvSourceProviderProof.from_dict(proof.to_dict())
    assert restored == proof
    assert proof.to_dict() == proof.to_dict()


def test_proof_enum_and_nested_serialization_shape() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    assert payload["provider_role"] == proof.provider_role.value
    assert payload["domain_kind"] == proof.domain_kind.value
    assert payload["pass_through_dimensions"] == []
    assert isinstance(payload["artifact_components"], list)
    assert isinstance(payload["header"], dict)
    assert payload["sample_index_domain_digest"] is None
    assert payload["canonical_point_content_digest"] is None


def test_proof_from_dict_rejects_missing_field() -> None:
    payload = _payload()
    del payload["provider_key"]
    with pytest.raises(ValueError, match="missing"):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_extra_field() -> None:
    payload = _payload()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_wrong_root_type() -> None:
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict("not-a-mapping")  # type: ignore[arg-type]


def test_proof_from_dict_rejects_wrong_nested_type() -> None:
    payload = _payload()
    payload["header"] = "not-a-header"
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)
    payload = _payload()
    payload["artifact_components"] = [{"component_key": "x"}]
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)
    payload = _payload()
    payload["origin_xyz"] = "not-a-list"
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_bool_as_int() -> None:
    payload = _payload()
    payload["source_point_count"] = True
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_nan_and_infinity() -> None:
    payload = _payload()
    payload["source_bounds_min_xyz"] = [0.0, 0.0, float("nan")]
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)
    payload = _payload()
    payload["source_bounds_max_xyz"] = [0.0, 0.0, float("inf")]
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)
    payload = _payload()
    payload["origin_xyz"] = [0.0, 0.0, float("-inf")]
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_invalid_enum() -> None:
    payload = _payload()
    payload["provider_role"] = "not_a_role"
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_invalid_digest() -> None:
    payload = _payload()
    payload["configuration_digest"] = "not-a-sha"
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_invalid_count() -> None:
    payload = _payload()
    payload["source_point_count"] = 0
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_from_dict_rejects_inconsistent_nullable_pair() -> None:
    payload = _payload()
    payload["canonical_point_content_digest"] = "b" * 64
    payload["canonical_point_content_spec_digest"] = None
    with pytest.raises(ValueError):
        source.FrvSourceProviderProof.from_dict(payload)


def test_proof_input_mutation_isolation() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    restored = source.FrvSourceProviderProof.from_dict(dict(payload))
    payload["configuration_digest"] = "b" * 64
    payload["artifact_components"][0]["sha256"] = "b" * 64
    payload["header"]["las_version"] = "9.9"
    assert restored.configuration_digest == SHA
    assert restored.artifact_components[0].sha256 == SHA
    assert restored.header.las_version == "1.2"


def test_proof_output_mutation_isolation() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    payload["configuration_digest"] = "b" * 64
    payload["artifact_components"][0]["sha256"] = "b" * 64
    payload["header"]["las_version"] = "9.9"
    assert proof.configuration_digest == SHA
    assert proof.artifact_components[0].sha256 == SHA
    assert proof.header.las_version == "1.2"


def test_result_has_no_serialization() -> None:
    result = _valid_result()
    assert not hasattr(result, "to_dict")
    assert not hasattr(result, "from_dict")


def test_capability_not_in_any_serialization() -> None:
    proof = _valid_proof()
    payload = proof.to_dict()
    assert "_finalized_capability" not in payload
    assert "capability" not in payload


def test_private_symbols_not_exported() -> None:
    assert not hasattr(contracts, "_FRV_SOURCE_FINALIZED_CAPABILITY")
    assert not hasattr(contracts, "_create_finalized_frv_source_provider_result")


def test_proof_with_canonical_content_round_trip() -> None:
    spec = _exact_canonical_spec()
    identity = _source_identity(
        path=r"C:\__frv_source_contract_test__\source.las",
        canonical_digest="b" * 64,
        canonical_spec_digest=spec.spec_digest,
        canonical_spec_version=frv.FRV_CANONICAL_POINT_CONTENT_SCHEMA_VERSION,
    )
    proof = _valid_proof(
        source_content_identity_digest=identity.source_content_identity_digest,
        source_identity_record_digest=identity.source_identity_record_digest,
        canonical_point_content_digest="b" * 64,
        canonical_point_content_spec_digest=spec.spec_digest,
    )
    restored = source.FrvSourceProviderProof.from_dict(proof.to_dict())
    assert restored == proof
    assert restored.canonical_point_content_digest == "b" * 64
    assert restored.canonical_point_content_spec_digest == spec.spec_digest


def test_frv_source_module_has_no_static_laspy_dependency() -> None:
    tree = ast.parse(Path(source.__file__).read_text(encoding="utf-8"))
    top_level_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top_level_modules.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module is not None:
                top_level_modules.append(node.module.split(".")[0])
    assert "laspy" not in top_level_modules, (
        "core.contracts.frv_source must not statically import laspy"
    )
