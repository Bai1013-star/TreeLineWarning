"""Focused FRV-to-V4 input contract tests (PLAN-FROZEN)."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import core.contracts.v4_inputs as v4
import numpy as np
import pytest
from core.contracts.frv import frv_digest
from core.contracts.v4_inputs import (
    V4_COVERAGE_DIGEST_DOMAIN,
    V4_HANDOFF_DIGEST_DOMAIN,
    V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2,
    V4_INPUT_MANIFEST_DIGEST_DOMAIN,
    V4_RECOVERED_INDEX_DIGEST_DOMAIN,
    V4InputArtifactManifest,
    V4InputErrorCode,
    V4InputHandoffResult,
    V4InputLoadError,
    V4InputOutputReference,
    V4PointRecoveryState,
    V4PrimaryRoute,
    V4RecoveredVegetationSet,
    V4RecoveryCoverageReport,
    V4UnrecoveredRegionRecord,
    V4UnresolvedMeasuredPointSet,
    frv_v4_handoff_digest,
    frv_v4_recovered_index_digest,
)

from tests.io.test_source_provider import _source_identity

SHA = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64


def _canon(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _source() -> object:
    return _source_identity("C:/s.las", SHA, 4, ([0.0, 0.0, 0.0], [3.0, 3.0, 10.0]))


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _recovered() -> V4RecoveredVegetationSet:
    return V4RecoveredVegetationSet(
        source_identity=_source(),
        original_index=_ro(np.asarray([0, 1, 2], dtype="<i8")),
        xyz_local_m=_ro(
            np.asarray([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 2.0, 3.0]], dtype="<f8")
        ),
        hag_m=_ro(np.asarray([1.0, 2.0, 3.0], dtype="<f4")),
        membership_state=_ro(np.asarray([1, 1, 1], dtype="|u1")),
        hag_evidence_state=_ro(np.asarray([1, 1, 1], dtype="|u1")),
        segmentation_hag_readiness=_ro(np.asarray([1, 1, 1], dtype="|u1")),
        candidate_reason_flags=_ro(np.zeros(3, dtype="<u4")),
        candidate_confidence=None,
        frv_membership_artifact_digest=SHA,
        recovered_index_digest=frv_v4_recovered_index_digest(
            source_content_identity_digest=SHA, original_index=[0, 1, 2]
        ),
    )


def test_recovered_index_digest_oracle() -> None:
    actual = frv_v4_recovered_index_digest(
        source_content_identity_digest=SHA, original_index=[0, 1, 2]
    )
    hasher = hashlib.sha256()
    hasher.update(b"V4RECIDX1\0")
    hasher.update(SHA.encode("utf-8") + b"\0")
    hasher.update(struct.pack("<Q", 3))
    for value in (0, 1, 2):
        hasher.update(struct.pack("<Q", value))
    assert actual == hasher.hexdigest()
    assert V4_RECOVERED_INDEX_DIGEST_DOMAIN == "V4RECIDX1"


def test_recovered_set_builds() -> None:
    recovered = _recovered()
    assert recovered.candidate_confidence is None
    assert recovered.original_index.dtype == np.dtype("<i8")


def test_recovered_set_rejects_writable() -> None:
    with pytest.raises(ValueError):
        V4RecoveredVegetationSet(
            source_identity=_source(),
            original_index=np.asarray([0, 1, 2], dtype="<i8"),
            xyz_local_m=_ro(np.zeros((3, 3), dtype="<f8")),
            hag_m=_ro(np.zeros(3, dtype="<f4")),
            membership_state=_ro(np.zeros(3, dtype="|u1")),
            hag_evidence_state=_ro(np.zeros(3, dtype="|u1")),
            segmentation_hag_readiness=_ro(np.zeros(3, dtype="|u1")),
            candidate_reason_flags=_ro(np.zeros(3, dtype="<u4")),
            candidate_confidence=None,
            frv_membership_artifact_digest=SHA,
            recovered_index_digest=SHA,
        )


def test_recovered_set_rejects_confidence() -> None:
    with pytest.raises(ValueError):
        V4RecoveredVegetationSet(
            source_identity=_source(),
            original_index=_ro(np.asarray([0], dtype="<i8")),
            xyz_local_m=_ro(np.zeros((1, 3), dtype="<f8")),
            hag_m=_ro(np.zeros(1, dtype="<f4")),
            membership_state=_ro(np.zeros(1, dtype="|u1")),
            hag_evidence_state=_ro(np.zeros(1, dtype="|u1")),
            segmentation_hag_readiness=_ro(np.zeros(1, dtype="|u1")),
            candidate_reason_flags=_ro(np.zeros(1, dtype="<u4")),
            candidate_confidence=0.5,  # type: ignore[arg-type]
            frv_membership_artifact_digest=SHA,
            recovered_index_digest=SHA,
        )


def test_unresolved_set_builds() -> None:
    count = 2
    unresolved = V4UnresolvedMeasuredPointSet(
        source_identity=_source(),
        original_index=_ro(np.asarray([0, 1], dtype="<i8")),
        xyz_local_m=_ro(np.zeros((count, 3), dtype="<f8")),
        membership_state=_ro(np.zeros(count, dtype="|u1")),
        hag_evidence_state=_ro(np.zeros(count, dtype="|u1")),
        segmentation_hag_readiness=_ro(np.zeros(count, dtype="|u1")),
        raw_hag_m=_ro(np.zeros(count, dtype="<f8")),
        clamped_hag_m=_ro(np.zeros(count, dtype="<f8")),
        reason_flags=_ro(np.zeros(count, dtype="<u4")),
        quality_flags=_ro(np.zeros(count, dtype="<u4")),
        wire_provider_state=_ro(np.zeros(count, dtype="|u1")),
        wire_relation_state=_ro(np.zeros(count, dtype="|u1")),
        nearest_wire_id=_ro(np.full(count, -1, dtype="<i4")),
        centerline_distance_m=_ro(np.zeros(count, dtype="<f8")),
        primary_route=_ro(np.full(count, int(V4PrimaryRoute.MEASURED_POINT_REVIEW), dtype="|u1")),
        frv_membership_artifact_digest=SHA,
        source_proof_digest=SHA,
        hag_proof_digest=SHA,
        geometry_proof_digest=SHA,
        wire_proof_digest=SHA,
        set_digest=SHA,
    )
    assert (unresolved.primary_route == int(V4PrimaryRoute.MEASURED_POINT_REVIEW)).all()


def _region(**overrides) -> V4UnrecoveredRegionRecord:
    defaults: dict = {
        "schema_version": v4.V4_UNRECOVERED_REGION_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "frv_scope_coverage_manifest_digest": _SHA_B,
        "canonical_inventory_key": "V1:00000000000000000001",
        "scope_id": None,
        "frv_scope_union_id": None,
        "scope_union_geometry_digest": None,
        "failure_reason_flags": 1024,
        "safety_relevance": "UNKNOWN",
        "review_required": True,
        "measured_clearance_complete": False,
        "diagnostic_reference": "coverage-blocked",
        "record_digest": None,
    }
    defaults.update(overrides)
    fields = {key: value for key, value in defaults.items() if key != "record_digest"}
    record_digest = (
        defaults["record_digest"]
        if defaults["record_digest"] is not None
        else hashlib.sha256(_canon(fields)).hexdigest()
    )
    return V4UnrecoveredRegionRecord(**{**fields, "record_digest": record_digest})


def test_unrecovered_region_digest_oracle() -> None:
    record = _region()
    expected = hashlib.sha256(_canon(record.to_digest_dict())).hexdigest()
    assert record.record_digest == expected


def test_unrecovered_region_rejections() -> None:
    with pytest.raises(ValueError):
        _region(measured_clearance_complete=True)
    with pytest.raises(ValueError):
        _region(safety_relevance="KNOWN")
    with pytest.raises(ValueError):
        _region(frv_scope_union_id="u", scope_union_geometry_digest=None)


def _coverage_report(**overrides) -> V4RecoveryCoverageReport:
    defaults: dict = {
        "schema_version": v4.V4_RECOVERY_COVERAGE_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "frv_scope_coverage_manifest_digest": _SHA_C,
        "frv_scope_union_id": "scope-union-1",
        "frv_upstream_inventory_digest": SHA,
        "frv_membership_artifact_digest": SHA,
        "recovered_index_digest": SHA,
        "frv_canonical_inventory_count": 0,
        "frv_required_inventory_count": 0,
        "frv_approved_hard_rejection_count": 0,
        "frv_scope_created_or_merged_count": 0,
        "frv_coverage_blocked_count": 0,
        "frv_unique_in_scope_point_count": 0,
        "frv_evaluated_point_count": 0,
        "frv_unasserted_in_scope_count": 0,
        "frv_verified_vegetation_candidate_count": 0,
        "frv_excluded_non_candidate_count": 0,
        "frv_review_required_membership_count": 0,
        "frv_manual_include_count": 0,
        "frv_manual_exclude_count": 0,
        "normal_segmentation_count": 0,
        "measured_point_review_count": 0,
        "excluded_audit_count": 0,
        "point_recovery_failed_count": 0,
        "unrecovered_region_count": 0,
        "frv_canonical_inventory_complete": True,
        "frv_required_inventory_complete": True,
        "frv_approved_hard_rejection_complete": True,
        "frv_scope_coverage_complete": True,
        "frv_routing_complete": True,
        "frv_membership_decision_complete": True,
        "source_identity_complete": True,
        "xyz_recovery_complete": True,
        "hag_ready_complete": True,
        "segmentation_input_complete": True,
        "v4_routing_complete": True,
        "measured_clearance_complete": False,
        "review_required": False,
        "failure_codes": (),
        "report_digest": None,
    }
    defaults.update(overrides)
    report = V4RecoveryCoverageReport.__new__(V4RecoveryCoverageReport)
    for name, value in defaults.items():
        if name != "report_digest":
            object.__setattr__(report, name, value)
    object.__setattr__(report, "report_digest", report.computed_digest())
    return report


def test_coverage_report_digest_oracle() -> None:
    report = _coverage_report()
    payload = {
        "digest_domain": V4_COVERAGE_DIGEST_DOMAIN,
        "schema_version": v4.V4_RECOVERY_COVERAGE_SCHEMA_VERSION,
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "frv_scope_coverage_manifest_digest": _SHA_C,
        "frv_scope_union_id": "scope-union-1",
        "frv_upstream_inventory_digest": SHA,
        "frv_membership_artifact_digest": SHA,
        "recovered_index_digest": SHA,
        **{name: getattr(report, name) for name in v4.__dict__["_REPORT_COUNT_FIELDS"]},
        **{name: getattr(report, name) for name in v4.__dict__["_REPORT_BOOL_FIELDS"]},
        "failure_codes": [],
    }
    expected = hashlib.sha256(_canon(payload)).hexdigest()
    assert report.report_digest == expected


def test_coverage_report_rejects_clearance_complete() -> None:
    report = _coverage_report(measured_clearance_complete=True)
    with pytest.raises(ValueError):
        v4.validate_frv_recovery_coverage_report(report)


def test_handoff_digest_framing_oracle() -> None:
    record = _region()
    actual = frv_v4_handoff_digest(
        source_content_identity_digest=SHA,
        frv_membership_artifact_digest=_SHA_B,
        report_digest=_SHA_C,
        normal_set_digest=SHA,
        review_set_digest=_SHA_B,
        unrecovered_regions=(record,),
    )
    hasher = hashlib.sha256()
    hasher.update(b"V4INP1\0")
    for digest in (SHA, _SHA_B, _SHA_C, SHA, _SHA_B):
        hasher.update(digest.encode("utf-8") + b"\0")
    hasher.update(struct.pack("<Q", 1))
    hasher.update(bytes.fromhex(record.record_digest))
    assert actual == hasher.hexdigest()
    assert V4_HANDOFF_DIGEST_DOMAIN == "V4INP1"


def test_handoff_result_coverage_only() -> None:
    report = _coverage_report()
    output = V4InputOutputReference(
        root_path=Path("root"),
        validated_view_json_path=Path("root/view.json"),
        coverage_json_path=Path("root/coverage.json"),
        normal_block_paths=(),
        review_block_paths=(),
        handoff_manifest_path=Path("root/handoff.json"),
    )
    result = V4InputHandoffResult(
        validated_membership=None,
        recovered_vegetation=None,
        unresolved_measured_points=None,
        unrecovered_regions=(),
        coverage_report=report,
        output_reference=output,
        handoff_digest=SHA,
    )
    assert result.coverage_report is report
    with pytest.raises(ValueError):
        V4InputHandoffResult(
            validated_membership=None,
            recovered_vegetation=_recovered(),
            unresolved_measured_points=None,
            unrecovered_regions=(),
            coverage_report=report,
            output_reference=None,
            handoff_digest=SHA,
        )


def _manifest(**overrides) -> V4InputArtifactManifest:
    fields = {
        "schema_version": V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2,
        "source_content_identity_digest": SHA,
        "source_identity_record_digest": _SHA_B,
        "origin_context": (500000.0, 4200000.0, 125.0),
        "frv_membership_artifact_digest": _SHA_C,
        "report_digest": SHA,
        "source_proof_digest": _SHA_B,
        "hag_proof_digest": _SHA_C,
        "geometry_proof_digest": SHA,
        "wire_proof_digest": _SHA_B,
        "normal_block_count": 2,
        "review_block_count": 1,
        "normal_row_count": 5,
        "review_row_count": 2,
        "normal_block_paths": (
            "normal/v4nrm-block-00000000.v4blk",
            "normal/v4nrm-block-00000001.v4blk",
        ),
        "review_block_paths": ("review/v4rev-block-00000000.v4blk",),
        "normal_block_digests": (SHA, _SHA_B),
        "review_block_digests": (_SHA_C,),
        "normal_set_digest": SHA,
        "review_set_digest": _SHA_B,
        "handoff_digest": _SHA_C,
        "manifest_digest": "0" * 64,
    }
    fields.update(overrides)
    manifest = V4InputArtifactManifest(
        **{key: value for key, value in fields.items() if key != "manifest_digest"},
        manifest_digest=frv_digest(
            {"digest_domain": V4_INPUT_MANIFEST_DIGEST_DOMAIN, **{
                key: value for key, value in fields.items() if key != "manifest_digest"
            }}
        ),
    )
    return manifest


def test_v2_manifest_digest_oracle() -> None:
    manifest = _manifest()
    expected = frv_digest(
        {
            "digest_domain": V4_INPUT_MANIFEST_DIGEST_DOMAIN,
            **{
                key: value
                for key, value in manifest.to_digest_dict().items()
                if key != "digest_domain"
            },
        }
    )
    assert manifest.manifest_digest == expected
    assert manifest.manifest_digest == manifest.computed_digest()
    assert V4_INPUT_MANIFEST_DIGEST_DOMAIN == "V4INPMF1"
    assert V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2 == "v4-input-handoff-v2"


def test_v2_manifest_round_trip() -> None:
    manifest = _manifest()
    parsed = V4InputArtifactManifest.from_dict(manifest.to_dict())
    assert parsed == manifest


def test_v2_manifest_rejects_bad_path_inventory() -> None:
    with pytest.raises(ValueError):
        _manifest(normal_block_paths=("normal/wrong-name.v4blk", "normal/x.v4blk"))
    with pytest.raises(ValueError):
        _manifest(normal_block_paths=("normal\\v4nrm-block-00000000.v4blk",))


def test_v2_manifest_rejects_count_mismatch() -> None:
    with pytest.raises(ValueError):
        _manifest(normal_block_count=3)


def test_v2_manifest_rejects_v1_schema() -> None:
    with pytest.raises(ValueError):
        _manifest(schema_version="v4-input-handoff-v1")


def test_error_codes_frozen() -> None:
    assert V4InputErrorCode.FRV_RESULT_NOT_READY.value == "FRV_RESULT_NOT_READY"
    assert V4InputErrorCode.OUTPUT_PUBLICATION_FAILED.value == "OUTPUT_PUBLICATION_FAILED"
    error = V4InputLoadError(
        code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED, field_name="partition"
    )
    assert error.code is V4InputErrorCode.ROUTING_CONSERVATION_FAILED
    assert error.field_name == "partition"


def test_point_recovery_state_frozen() -> None:
    assert V4PointRecoveryState.POINT_RECOVERY_FAILED == 1
    assert V4PointRecoveryState.POINT_RECOVERED_HAG_READY == 2
    assert V4PointRecoveryState.POINT_RECOVERED_HAG_REVIEW_REQUIRED == 3
    assert V4PointRecoveryState.POINT_RECOVERED_HAG_MISSING == 4
    assert V4PointRecoveryState.POINT_RECOVERED_HAG_INVALID == 5
    assert V4PrimaryRoute.NORMAL_SEGMENTATION == 1
    assert V4PrimaryRoute.MEASURED_POINT_REVIEW == 2
    assert V4PrimaryRoute.UNRECOVERED_REGION_REVIEW == 3
    assert V4PrimaryRoute.EXCLUDED_AUDIT_ONLY == 4
