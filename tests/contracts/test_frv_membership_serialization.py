"""Exact FRV membership serialization tests (PLAN-FROZEN)."""

from __future__ import annotations

import dataclasses
import json

import pytest
from core.contracts.frv_membership import (
    FrvDecisionEvidenceArtifact,
    FrvMembershipExecutionResult,
    FrvMembershipPolicyConfig,
    FrvScopePolicyConfig,
)

from tests.contracts.test_frv_membership_contracts import (
    _evidence,
    _result,
)
from tests.contracts.test_frv_membership_contracts import (
    default_frv_membership_policy_config as _mem_policy,
)
from tests.contracts.test_frv_membership_contracts import (
    default_frv_scope_policy_config as _scope_policy,
)


def test_scope_policy_serialization_exact_shape() -> None:
    config = _scope_policy()
    payload = config.to_dict()
    assert list(payload) == [field.name for field in dataclasses.fields(FrvScopePolicyConfig)]
    restored = FrvScopePolicyConfig.from_dict(payload)
    assert restored == config
    assert json.loads(json.dumps(payload)) == payload


def test_membership_policy_serialization_exact_shape() -> None:
    config = _mem_policy()
    payload = config.to_dict()
    assert list(payload) == [
        field.name for field in dataclasses.fields(FrvMembershipPolicyConfig)
    ]
    restored = FrvMembershipPolicyConfig.from_dict(payload)
    assert restored == config


def test_scope_policy_serialization_rejects_missing_extra() -> None:
    payload = _scope_policy().to_dict()
    del payload["voxel_size_m"]
    with pytest.raises(ValueError, match="missing"):
        FrvScopePolicyConfig.from_dict(payload)
    payload = _scope_policy().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvScopePolicyConfig.from_dict(payload)


def test_decision_evidence_serialization_exact_shape() -> None:
    artifact = _evidence()
    payload = artifact.to_dict()
    assert list(payload) == [
        field.name for field in dataclasses.fields(FrvDecisionEvidenceArtifact)
    ]
    assert "digest_domain" not in payload
    restored = FrvDecisionEvidenceArtifact.from_dict(payload)
    assert restored == artifact


def test_decision_evidence_serialization_rejects_missing_extra() -> None:
    payload = _evidence().to_dict()
    del payload["record_count"]
    with pytest.raises(ValueError, match="missing"):
        FrvDecisionEvidenceArtifact.from_dict(payload)
    payload = _evidence().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvDecisionEvidenceArtifact.from_dict(payload)


def test_execution_result_serialization_round_trip_via_json() -> None:
    result = _result()
    payload = result.to_dict()
    assert "output_reference" not in payload
    assert "result_digest" in payload
    encoded = json.dumps(payload, ensure_ascii=False)
    restored = FrvMembershipExecutionResult.from_dict(json.loads(encoded))
    assert restored.to_dict() == result.to_dict()
    assert restored.output_reference is None


def test_execution_result_serialization_rejects_unknown() -> None:
    payload = _result().to_dict()
    payload["extra"] = 1
    with pytest.raises(ValueError, match="unknown"):
        FrvMembershipExecutionResult.from_dict(payload)
