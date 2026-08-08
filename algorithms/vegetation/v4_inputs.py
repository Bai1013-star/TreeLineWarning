"""FRV-to-V4 input handoff runtime (PLAN-FROZEN).

``build_v4_input_handoff`` consumes a formal membership execution result: it
re-validates the published FRV files, opens exactly one P-SOURCE session to
merge-recover the original XYZ for the recovery domain, partitions the exact
normal/review/excluded routes, publishes the V4 normal and review blocks, the
validated view, the recovery-coverage report and the handoff manifest with the
atomic temp-directory validate/rename discipline.  A COVERAGE_BLOCKED result
produces a coverage-only handoff; a NO_SCOPE result raises
``V4InputLoadError(FRV_RESULT_NOT_READY)``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import laspy
import numpy as np
from core.contracts.frv import (
    FrvConfiguration,
    FrvMembershipState,
    FrvSegmentationHagReadiness,
    SourceIdentity,
    frv_digest,
)
from core.contracts.frv_membership import (
    FrvMembershipExecutionResult,
    FrvMembershipExecutionState,
    frv_source_proof_digest,
)
from core.contracts.frv_source import (
    FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
    FrvSourceCanonicalContentPolicy,
    FrvSourceCoordinateConversionPolicy,
    FrvSourceInvalidPointPolicy,
    FrvSourceModel,
    FrvSourcePassThroughDimension,
    FrvSourceProviderConfig,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
    derive_frv_source_provider_config_digest,
    validate_finalized_frv_source_provider_result,
)
from core.contracts.v4_inputs import (
    V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2,
    V4_INPUT_MANIFEST_DIGEST_DOMAIN,
    V4_RECOVERY_COVERAGE_SCHEMA_VERSION,
    V4_UNRECOVERED_REGION_SCHEMA_VERSION,
    V4InputArtifactManifest,
    V4InputErrorCode,
    V4InputHandoffResult,
    V4InputLoadError,
    V4InputOutputReference,
    V4PrimaryRoute,
    V4RecoveredVegetationSet,
    V4RecoveryCoverageReport,
    V4UnrecoveredRegionRecord,
    V4UnresolvedMeasuredPointSet,
    V4ValidatedFrvMembershipView,
    frv_v4_handoff_digest,
    frv_v4_recovered_index_digest,
    validate_frv_recovery_coverage_report,
    validate_frv_validated_membership_view,
)

from algorithms.io.source_provider import load_frv_source
from algorithms.vegetation.frv_artifacts import (
    FRV_BLOCK_MAGIC,
    FRV_EVIDENCE_BLOCK_MAGIC,
    block_id_for_ordinal,
    decode_frv_block,
    encode_frv_block,
    publish_directory,
    write_canonical_json,
)

# V4 block magic constants.
V4_NORMAL_BLOCK_MAGIC = b"V4NRMBLK1\0"
V4_REVIEW_BLOCK_MAGIC = b"V4REVBLK1\0"

# Frozen V4 block field inventories.
_NORMAL_FIELDS = (
    ("original_index", "<i8"),
    ("xyz_local_m", "<f8"),
    ("hag_m", "<f4"),
    ("membership_state", "|u1"),
    ("hag_evidence_state", "|u1"),
    ("segmentation_hag_readiness", "|u1"),
    ("candidate_reason_flags", "<u4"),
)
_REVIEW_FIELDS = (
    ("original_index", "<i8"),
    ("xyz_local_m", "<f8"),
    ("membership_state", "|u1"),
    ("hag_evidence_state", "|u1"),
    ("segmentation_hag_readiness", "|u1"),
    ("raw_hag_m", "<f8"),
    ("clamped_hag_m", "<f8"),
    ("reason_flags", "<u4"),
    ("quality_flags", "<u4"),
    ("wire_provider_state", "|u1"),
    ("wire_relation_state", "|u1"),
    ("nearest_wire_id", "<i4"),
    ("centerline_distance_m", "<f8"),
    ("primary_route", "|u1"),
)

_V4_BLOCK_CAPACITY = 250000

# Wire provider/relation states needed for the review set (stored in evidence).
_WIRE_NOT_APPLICABLE_PROVEN = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_v4_input_handoff(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    membership_result: FrvMembershipExecutionResult,
    *,
    frv_configuration: FrvConfiguration,
    output_dir: Path,
    source_chunk_size: int = 1000000,
) -> V4InputHandoffResult:
    """Build the V4 input handoff from a formal membership execution result."""

    _check_signature_types(
        reference, source_result, membership_result, output_dir, source_chunk_size
    )
    state = membership_result.execution_state
    if state is FrvMembershipExecutionState.NO_RELEVANT_FRV_SCOPE:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_RESULT_NOT_READY, field_name="execution_state"
        )
    if state is FrvMembershipExecutionState.COVERAGE_BLOCKED:
        return _coverage_only_handoff(membership_result)

    # READY path.
    if membership_result.output_reference is None:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="output_reference"
        )
    from algorithms.vegetation.frv_membership import load_frv_membership_result

    root_path = membership_result.output_reference.root_path
    reloaded = load_frv_membership_result(root_path)
    if reloaded.execution_state is not FrvMembershipExecutionState.READY_FOR_V4:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="execution_state"
        )

    payload_rows, evidence_rows = _read_artifact_rows(membership_result)
    evaluated_indexes = sorted(payload_rows)
    _verify_evaluated_indexes(payload_rows, evidence_rows, reloaded)

    recovery_targets = [
        index
        for index in evaluated_indexes
        if int(payload_rows[index]["membership_state"])
        not in (
            int(FrvMembershipState.EXCLUDED_NON_CANDIDATE),
            int(FrvMembershipState.MANUAL_EXCLUDE),
        )
    ]
    recovered_xyz = _recover_xyz(
        reference,
        source_result,
        frv_configuration,
        recovery_targets,
        source_chunk_size,
        expected_source_proof_digest=membership_result.source_proof_digest,
    )
    if recovered_xyz is None:
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_INVALID, field_name="original_index"
        )

    normal_indexes: list[int] = []
    review_indexes: list[int] = []
    for index in recovery_targets:
        row = payload_rows[index]
        state_value = int(row["membership_state"])
        readiness = int(row["segmentation_hag_readiness"])
        admitted = state_value in (
            int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE),
            int(FrvMembershipState.MANUAL_INCLUDE),
        )
        if admitted and readiness == int(FrvSegmentationHagReadiness.READY):
            normal_indexes.append(index)
        else:
            review_indexes.append(index)

    normal_indexes.sort()
    review_indexes.sort()
    # Canonical local XYZ: ``xyz_local_m = source_xyz - origin_context`` (frozen
    # plan 9.2).  The replay returns absolute measured XYZ; the origin is applied
    # exactly once here so in-memory sets and published normal/review blocks all
    # share the same local coordinate source.
    origin_context = tuple(float(v) for v in reloaded.source_identity.origin_context)
    local_xyz = {
        index: recovered_xyz[index] - np.asarray(origin_context, dtype=np.float64)
        for index in recovered_xyz
    }
    normal_xyz = {index: local_xyz[index] for index in normal_indexes}
    review_xyz = {index: local_xyz[index] for index in review_indexes}

    recovered_set = _build_recovered_set(
        reloaded, normal_indexes, normal_xyz, payload_rows
    )
    unresolved_set = _build_unresolved_set(
        reloaded, review_indexes, review_xyz, payload_rows, evidence_rows
    )
    coverage_report = _build_ready_coverage_report(
        reloaded,
        payload_rows,
        normal_indexes,
        review_indexes,
        recovered_set,
        unresolved_set,
    )

    view = _build_validated_view(membership_result, reloaded)
    handoff_reference, manifest = _publish_ready_handoff(
        output_dir=output_dir,
        reference_dir=root_path,
        view=view,
        coverage_report=coverage_report,
        recovered_set=recovered_set,
        unresolved_set=unresolved_set,
        normal_indexes=normal_indexes,
        review_indexes=review_indexes,
        normal_xyz=normal_xyz,
        review_xyz=review_xyz,
        payload_rows=payload_rows,
        evidence_rows=evidence_rows,
        membership_artifact_digest=reloaded.membership_artifact.manifest_digest,
        source_proof_digest=reloaded.source_proof_digest,
        hag_proof_digest=reloaded.hag_proof_digest or ("0" * 64),
        geometry_proof_digest=reloaded.geometry_proof_digest or ("0" * 64),
        wire_proof_digest=reloaded.wire_proof_digest or ("0" * 64),
    )
    return _make_handoff(
        validated_membership=view,
        recovered_vegetation=recovered_set,
        unresolved_measured_points=unresolved_set,
        unrecovered_regions=(),
        coverage_report=coverage_report,
        output_reference=handoff_reference,
        handoff_digest=manifest.handoff_digest,
    )


# ---------------------------------------------------------------------------
# Signature and state helpers
# ---------------------------------------------------------------------------


def _check_signature_types(
    reference: Any,
    source_result: Any,
    membership_result: Any,
    output_dir: Any,
    source_chunk_size: Any,
) -> None:
    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(membership_result) is not FrvMembershipExecutionResult:
        raise TypeError("membership_result must be a FrvMembershipExecutionResult")
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    if type(source_chunk_size) is not int or source_chunk_size <= 0:
        raise ValueError("source_chunk_size must be a positive built-in int")


def _make_handoff(*args: Any, **kwargs: Any) -> V4InputHandoffResult:
    # Construct through the real constructor so the full handoff invariants
    # (_check_handoff_result) run on every reconstructed result; a __new__
    # object is never trusted without validation.
    return V4InputHandoffResult(
        validated_membership=kwargs["validated_membership"],
        recovered_vegetation=kwargs["recovered_vegetation"],
        unresolved_measured_points=kwargs["unresolved_measured_points"],
        unrecovered_regions=kwargs["unrecovered_regions"],
        coverage_report=kwargs["coverage_report"],
        output_reference=kwargs["output_reference"],
        handoff_digest=kwargs["handoff_digest"],
    )


# ---------------------------------------------------------------------------
# Artifact re-read
# ---------------------------------------------------------------------------


def _read_artifact_rows(
    membership_result: FrvMembershipExecutionResult,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, dict[str, np.ndarray]]]:
    artifact = membership_result.membership_artifact
    schema = membership_result.payload_schema
    output = membership_result.output_reference
    assert artifact is not None and schema is not None and output is not None

    schema_order = tuple(array.field_name for array in schema.arrays)
    dtypes = {array.field_name: np.dtype(array.dtype_code) for array in schema.arrays}
    payload_rows: dict[int, dict[str, np.ndarray]] = {}
    for block, block_path in zip(
        artifact.block_descriptors, output.block_paths, strict=True
    ):
        data = block_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != block.block_digest:
            raise V4InputLoadError(
                code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="block"
            )
        _header, arrays = decode_frv_block(
            data,
            magic=FRV_BLOCK_MAGIC,
            schema_order=schema_order,
            dtypes=dtypes,
            row_shapes={name: () for name in schema_order},
        )
        for position, index in enumerate(np.asarray(arrays["original_index"], dtype=np.int64)):
            payload_rows[int(index)] = {name: arrays[name][position] for name in schema_order}

    evidence_order = (
        "original_index",
        "geometry_evidence_state",
        "geometry_evidence_flags",
        "wire_provider_state",
        "wire_relation_state",
        "nearest_wire_id",
        "nearest_segment_index",
        "centerline_distance_m",
    )
    evidence_dtypes = {
        "original_index": np.dtype("<i8"),
        "geometry_evidence_state": np.dtype("|u1"),
        "geometry_evidence_flags": np.dtype("<u4"),
        "wire_provider_state": np.dtype("|u1"),
        "wire_relation_state": np.dtype("|u1"),
        "nearest_wire_id": np.dtype("<i4"),
        "nearest_segment_index": np.dtype("<i4"),
        "centerline_distance_m": np.dtype("<f8"),
    }
    evidence_rows: dict[int, dict[str, np.ndarray]] = {}
    for block_path in output.decision_evidence_block_paths:
        data = block_path.read_bytes()
        _header, arrays = decode_frv_block(
            data,
            magic=FRV_EVIDENCE_BLOCK_MAGIC,
            schema_order=evidence_order,
            dtypes=evidence_dtypes,
            row_shapes={name: () for name in evidence_order},
        )
        for position, index in enumerate(np.asarray(arrays["original_index"], dtype=np.int64)):
            evidence_rows[int(index)] = {name: arrays[name][position] for name in evidence_order}
    return payload_rows, evidence_rows


def _verify_evaluated_indexes(
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
    reloaded: FrvMembershipExecutionResult,
) -> None:
    if len(payload_rows) != reloaded.membership_artifact.evaluated_point_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="evaluated_point_count"
        )
    if set(payload_rows) != set(evidence_rows):
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="evidence_indexes"
        )
    indexes = sorted(payload_rows)
    if len(indexes) != len(set(indexes)):
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_DUPLICATE, field_name="original_index"
        )
    count = reloaded.source_identity.source_point_count
    if any(index < 0 or index >= count for index in indexes):
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_INVALID, field_name="original_index"
        )


# ---------------------------------------------------------------------------
# Source replay
# ---------------------------------------------------------------------------


def _recover_xyz(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    configuration: FrvConfiguration,
    targets: list[int],
    source_chunk_size: int,
    expected_source_proof_digest: str,
) -> dict[int, np.ndarray] | None:
    """Recover XYZ for *targets* with exactly one P-SOURCE pass.

    The replayed session must naturally exhaust, finalize and reproduce the
    canonical P-SOURCE proof fingerprint bound by the membership result
    (DC-001-SOURCE-PROOF-REBIND).  A structurally invalid or fingerprint
    mismatched replay fails closed before any recovered XYZ is used; ``None``
    is returned only when the recovered index set does not cover the targets.
    """

    target_set = set(targets)
    recovered: dict[int, np.ndarray] = {}
    source_config = _v4_source_provider_config()
    try:
        with load_frv_source(
            reference,
            frv_configuration=configuration,
            provider_config=source_config,
            chunk_size=source_chunk_size,
        ) as session:
            for chunk in session.iter_chunks():
                for row in range(chunk.point_count):
                    index = int(chunk.original_index[row])
                    if index in target_set:
                        recovered[index] = chunk.xyz[row].copy()
            replayed = session.finalize()
    except Exception:
        return None
    # Re-validate and compare the canonical source fingerprint before any
    # recovered XYZ is used by the handoff.
    try:
        validate_finalized_frv_source_provider_result(replayed)
        replayed_fingerprint = frv_source_proof_digest(replayed)
    except Exception as exc:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID,
            field_name="source_replay",
        ) from exc
    if replayed_fingerprint != expected_source_proof_digest:
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_replay",
        )
    if set(recovered) != target_set:
        return None
    return recovered


def _v4_source_provider_config() -> FrvSourceProviderConfig:
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,),
        provider_config_digest=derive_frv_source_provider_config_digest(
            schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
            source_model=FrvSourceModel.SINGLE_FILE_ONLY,
            coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
            invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
            canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
            pass_through_dimensions=(FrvSourcePassThroughDimension.CLASSIFICATION,),
        ),
    )


# ---------------------------------------------------------------------------
# Set construction
# ---------------------------------------------------------------------------


def _build_recovered_set(
    reloaded: FrvMembershipExecutionResult,
    normal_indexes: list[int],
    normal_xyz: dict[int, np.ndarray],
    payload_rows: Mapping[int, dict[str, np.ndarray]],
) -> V4RecoveredVegetationSet:
    count = len(normal_indexes)
    xyz = np.zeros((count, 3), dtype=np.float64)
    for position, index in enumerate(normal_indexes):
        xyz[position, :] = normal_xyz[index]
    set_obj = V4RecoveredVegetationSet.__new__(V4RecoveredVegetationSet)
    object.__setattr__(set_obj, "source_identity", reloaded.source_identity)
    object.__setattr__(set_obj, "original_index", _ro(np.asarray(normal_indexes, dtype="<i8")))
    object.__setattr__(set_obj, "xyz_local_m", _ro(xyz))
    object.__setattr__(
        set_obj,
        "hag_m",
        _ro(
            np.asarray(
                [float(payload_rows[index]["clamped_hag"]) for index in normal_indexes],
                dtype="<f4",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "membership_state",
        _ro(
            np.asarray(
                [int(payload_rows[index]["membership_state"]) for index in normal_indexes],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "hag_evidence_state",
        _ro(
            np.asarray(
                [int(payload_rows[index]["hag_evidence_state"]) for index in normal_indexes],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "segmentation_hag_readiness",
        _ro(
            np.asarray(
                [
                    int(payload_rows[index]["segmentation_hag_readiness"])
                    for index in normal_indexes
                ],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "candidate_reason_flags",
        _ro(
            np.asarray(
                [int(payload_rows[index]["reason_flags"]) for index in normal_indexes],
                dtype="<u4",
            )
        ),
    )
    object.__setattr__(set_obj, "candidate_confidence", None)
    object.__setattr__(
        set_obj,
        "frv_membership_artifact_digest",
        reloaded.membership_artifact.manifest_digest,
    )
    object.__setattr__(
        set_obj,
        "recovered_index_digest",
        frv_v4_recovered_index_digest(
            source_content_identity_digest=reloaded.source_identity.source_content_identity_digest,
            original_index=normal_indexes,
        ),
    )
    return set_obj


def _build_unresolved_set(
    reloaded: FrvMembershipExecutionResult,
    review_indexes: list[int],
    review_xyz: dict[int, np.ndarray],
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
) -> V4UnresolvedMeasuredPointSet:
    count = len(review_indexes)
    xyz = np.zeros((count, 3), dtype=np.float64)
    for position, index in enumerate(review_indexes):
        xyz[position, :] = review_xyz[index]
    block_digests = _review_block_digests(review_indexes, xyz, payload_rows, evidence_rows)
    set_digest = frv_digest(
        {"digest_domain": "V4REV1", "block_digests": list(block_digests)}
    )
    set_obj = V4UnresolvedMeasuredPointSet.__new__(V4UnresolvedMeasuredPointSet)
    object.__setattr__(set_obj, "source_identity", reloaded.source_identity)
    object.__setattr__(set_obj, "original_index", _ro(np.asarray(review_indexes, dtype="<i8")))
    object.__setattr__(set_obj, "xyz_local_m", _ro(xyz))
    object.__setattr__(
        set_obj,
        "membership_state",
        _ro(
            np.asarray(
                [int(payload_rows[index]["membership_state"]) for index in review_indexes],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "hag_evidence_state",
        _ro(
            np.asarray(
                [int(payload_rows[index]["hag_evidence_state"]) for index in review_indexes],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "segmentation_hag_readiness",
        _ro(
            np.asarray(
                [
                    int(payload_rows[index]["segmentation_hag_readiness"])
                    for index in review_indexes
                ],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "raw_hag_m",
        _ro(
            np.asarray(
                [float(payload_rows[index]["raw_hag"]) for index in review_indexes], dtype="<f8"
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "clamped_hag_m",
        _ro(
            np.asarray(
                [float(payload_rows[index]["clamped_hag"]) for index in review_indexes],
                dtype="<f8",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "reason_flags",
        _ro(
            np.asarray(
                [int(payload_rows[index]["reason_flags"]) for index in review_indexes], dtype="<u4"
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "quality_flags",
        _ro(
            np.asarray(
                [int(payload_rows[index]["input_quality_flags"]) for index in review_indexes],
                dtype="<u4",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "wire_provider_state",
        _ro(
            np.asarray(
                [int(evidence_rows[index]["wire_provider_state"]) for index in review_indexes],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "wire_relation_state",
        _ro(
            np.asarray(
                [int(evidence_rows[index]["wire_relation_state"]) for index in review_indexes],
                dtype="|u1",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "nearest_wire_id",
        _ro(
            np.asarray(
                [int(evidence_rows[index]["nearest_wire_id"]) for index in review_indexes],
                dtype="<i4",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "centerline_distance_m",
        _ro(
            np.asarray(
                [
                    float(evidence_rows[index]["centerline_distance_m"])
                    for index in review_indexes
                ],
                dtype="<f8",
            )
        ),
    )
    object.__setattr__(
        set_obj,
        "primary_route",
        _ro(np.full(count, int(V4PrimaryRoute.MEASURED_POINT_REVIEW), dtype="|u1")),
    )
    object.__setattr__(
        set_obj,
        "frv_membership_artifact_digest",
        reloaded.membership_artifact.manifest_digest,
    )
    object.__setattr__(
        set_obj, "source_proof_digest", reloaded.source_proof_digest
    )
    object.__setattr__(
        set_obj,
        "hag_proof_digest",
        reloaded.hag_proof_digest or ("0" * 64),
    )
    object.__setattr__(
        set_obj,
        "geometry_proof_digest",
        reloaded.geometry_proof_digest or ("0" * 64),
    )
    object.__setattr__(
        set_obj, "wire_proof_digest", reloaded.wire_proof_digest or ("0" * 64)
    )
    object.__setattr__(set_obj, "set_digest", set_digest)
    return set_obj


def _review_block_digests(
    review_indexes: list[int],
    xyz: np.ndarray,
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
) -> tuple[str, ...]:
    data = _encode_review_blocks(review_indexes, xyz, payload_rows, evidence_rows)
    return tuple(hashlib.sha256(block).hexdigest() for block in data)


def _encode_review_blocks(
    review_indexes: list[int],
    xyz: np.ndarray,
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
) -> list[bytes]:
    blocks: list[bytes] = []
    count = len(review_indexes)
    start = 0
    while start < count:
        stop = min(start + _V4_BLOCK_CAPACITY, count)
        arrays = _review_block_arrays(
            review_indexes[start:stop], xyz[start:stop], payload_rows, evidence_rows
        )
        blocks.append(
            encode_frv_block(
                magic=V4_REVIEW_BLOCK_MAGIC,
                artifact_id="v4-input-handoff",
                block_id=block_id_for_ordinal(len(blocks), evidence=False),
                payload_schema_digest="0" * 64,
                arrays=arrays,
                schema_order=tuple(name for name, _ in _REVIEW_FIELDS),
            )
        )
        start = stop
    if not blocks:
        blocks.append(
            encode_frv_block(
                magic=V4_REVIEW_BLOCK_MAGIC,
                artifact_id="v4-input-handoff",
                block_id="v4rev-block-00000000",
                payload_schema_digest="0" * 64,
                arrays=_empty_review_arrays(),
                schema_order=tuple(name for name, _ in _REVIEW_FIELDS),
            )
        )
    return blocks


def _review_block_arrays(
    indexes: list[int],
    xyz_slice: np.ndarray,
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "original_index": np.asarray(indexes, dtype="<i8"),
        "xyz_local_m": xyz_slice,
        "membership_state": np.asarray(
            [int(payload_rows[index]["membership_state"]) for index in indexes], dtype="|u1"
        ),
        "hag_evidence_state": np.asarray(
            [int(payload_rows[index]["hag_evidence_state"]) for index in indexes], dtype="|u1"
        ),
        "segmentation_hag_readiness": np.asarray(
            [int(payload_rows[index]["segmentation_hag_readiness"]) for index in indexes],
            dtype="|u1",
        ),
        "raw_hag_m": np.asarray(
            [float(payload_rows[index]["raw_hag"]) for index in indexes], dtype="<f8"
        ),
        "clamped_hag_m": np.asarray(
            [float(payload_rows[index]["clamped_hag"]) for index in indexes], dtype="<f8"
        ),
        "reason_flags": np.asarray(
            [int(payload_rows[index]["reason_flags"]) for index in indexes], dtype="<u4"
        ),
        "quality_flags": np.asarray(
            [int(payload_rows[index]["input_quality_flags"]) for index in indexes], dtype="<u4"
        ),
        "wire_provider_state": np.asarray(
            [int(evidence_rows[index]["wire_provider_state"]) for index in indexes], dtype="|u1"
        ),
        "wire_relation_state": np.asarray(
            [int(evidence_rows[index]["wire_relation_state"]) for index in indexes], dtype="|u1"
        ),
        "nearest_wire_id": np.asarray(
            [int(evidence_rows[index]["nearest_wire_id"]) for index in indexes], dtype="<i4"
        ),
        "centerline_distance_m": np.asarray(
            [float(evidence_rows[index]["centerline_distance_m"]) for index in indexes],
            dtype="<f8",
        ),
        "primary_route": np.full(
            len(indexes), int(V4PrimaryRoute.MEASURED_POINT_REVIEW), dtype="|u1"
        ),
    }
    for _name, array in arrays.items():
        array.flags.writeable = False
    return arrays


def _empty_review_arrays() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "original_index": np.empty(0, dtype="<i8"),
        "xyz_local_m": np.empty((0, 3), dtype="<f8"),
        "membership_state": np.empty(0, dtype="|u1"),
        "hag_evidence_state": np.empty(0, dtype="|u1"),
        "segmentation_hag_readiness": np.empty(0, dtype="|u1"),
        "raw_hag_m": np.empty(0, dtype="<f8"),
        "clamped_hag_m": np.empty(0, dtype="<f8"),
        "reason_flags": np.empty(0, dtype="<u4"),
        "quality_flags": np.empty(0, dtype="<u4"),
        "wire_provider_state": np.empty(0, dtype="|u1"),
        "wire_relation_state": np.empty(0, dtype="|u1"),
        "nearest_wire_id": np.empty(0, dtype="<i4"),
        "centerline_distance_m": np.empty(0, dtype="<f8"),
        "primary_route": np.empty(0, dtype="|u1"),
    }
    for _name, array in arrays.items():
        array.flags.writeable = False
    return arrays


def _ro(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


# ---------------------------------------------------------------------------
# Validated view
# ---------------------------------------------------------------------------


def _build_validated_view(
    membership_result: FrvMembershipExecutionResult,
    reloaded: FrvMembershipExecutionResult,
) -> V4ValidatedFrvMembershipView:
    artifact = membership_result.membership_artifact
    output = membership_result.output_reference
    assert artifact is not None and output is not None
    payload = {
        "source_identity": reloaded.source_identity,
        "frv_configuration_digest": artifact.configuration_digest,
        "coverage_manifest": reloaded.coverage_manifest,
        "membership_artifact": artifact,
        "payload_schema": reloaded.payload_schema,
        "artifact_binding": reloaded.artifact_binding,
        "decision_evidence_artifact": reloaded.decision_evidence_artifact,
        "membership_block_references": output.block_paths,
        "decision_evidence_block_references": output.decision_evidence_block_paths,
        "frv_canonical_inventory_count": artifact.canonical_inventory_count,
        "frv_required_inventory_count": artifact.required_inventory_count,
        "frv_approved_hard_rejection_count": artifact.approved_hard_rejection_count,
        "frv_scope_created_or_merged_count": artifact.scope_created_or_merged_count,
        "frv_coverage_blocked_count": artifact.coverage_blocked_count,
        "frv_unique_in_scope_point_count": artifact.unique_in_scope_point_count,
        "frv_evaluated_point_count": artifact.evaluated_point_count,
        "frv_unasserted_in_scope_count": artifact.unasserted_in_scope_count,
        "frv_verified_vegetation_candidate_count": artifact.verified_vegetation_candidate_count,
        "frv_excluded_non_candidate_count": artifact.excluded_non_candidate_count,
        "frv_review_required_membership_count": artifact.review_required_membership_count,
        "frv_manual_include_count": artifact.manual_include_count,
        "frv_manual_exclude_count": artifact.manual_exclude_count,
        "frv_canonical_inventory_complete": artifact.canonical_inventory_complete,
        "frv_required_inventory_complete": artifact.required_inventory_complete,
        "frv_approved_hard_rejection_complete": artifact.approved_hard_rejection_complete,
        "frv_scope_coverage_complete": artifact.scope_coverage_complete,
        "frv_routing_complete": artifact.routing_complete,
        "frv_membership_decision_complete": artifact.membership_decision_complete,
    }
    view = V4ValidatedFrvMembershipView.__new__(V4ValidatedFrvMembershipView)
    for name, value in payload.items():
        object.__setattr__(view, name, value)
    object.__setattr__(view, "view_digest", view.computed_digest())
    return view


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------


def _build_ready_coverage_report(
    reloaded: FrvMembershipExecutionResult,
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    normal_indexes: list[int],
    review_indexes: list[int],
    recovered_set: V4RecoveredVegetationSet,
    unresolved_set: V4UnresolvedMeasuredPointSet,
) -> V4RecoveryCoverageReport:
    artifact = reloaded.membership_artifact
    manifest = reloaded.coverage_manifest
    assert artifact is not None
    excluded_count = artifact.excluded_non_candidate_count + artifact.manual_exclude_count
    normal_count = len(normal_indexes)
    review_count = len(review_indexes)
    total = normal_count + review_count + excluded_count
    hag_ready_complete = _all_normal_ready(normal_indexes, payload_rows)
    values = {
        "schema_version": V4_RECOVERY_COVERAGE_SCHEMA_VERSION,
        "source_content_identity_digest": reloaded.source_identity.source_content_identity_digest,
        "source_identity_record_digest": reloaded.source_identity.source_identity_record_digest,
        "frv_scope_coverage_manifest_digest": manifest.manifest_digest,
        "frv_scope_union_id": artifact.scope_union_id,
        "frv_upstream_inventory_digest": manifest.canonical_inventory_digest,
        "frv_membership_artifact_digest": artifact.manifest_digest,
        "recovered_index_digest": recovered_set.recovered_index_digest,
        "frv_canonical_inventory_count": artifact.canonical_inventory_count,
        "frv_required_inventory_count": artifact.required_inventory_count,
        "frv_approved_hard_rejection_count": artifact.approved_hard_rejection_count,
        "frv_scope_created_or_merged_count": artifact.scope_created_or_merged_count,
        "frv_coverage_blocked_count": artifact.coverage_blocked_count,
        "frv_unique_in_scope_point_count": artifact.unique_in_scope_point_count,
        "frv_evaluated_point_count": artifact.evaluated_point_count,
        "frv_unasserted_in_scope_count": artifact.unasserted_in_scope_count,
        "frv_verified_vegetation_candidate_count": artifact.verified_vegetation_candidate_count,
        "frv_excluded_non_candidate_count": artifact.excluded_non_candidate_count,
        "frv_review_required_membership_count": artifact.review_required_membership_count,
        "frv_manual_include_count": artifact.manual_include_count,
        "frv_manual_exclude_count": artifact.manual_exclude_count,
        "normal_segmentation_count": normal_count,
        "measured_point_review_count": review_count,
        "excluded_audit_count": excluded_count,
        "point_recovery_failed_count": 0,
        "unrecovered_region_count": 0,
        "frv_canonical_inventory_complete": artifact.canonical_inventory_complete,
        "frv_required_inventory_complete": artifact.required_inventory_complete,
        "frv_approved_hard_rejection_complete": artifact.approved_hard_rejection_complete,
        "frv_scope_coverage_complete": artifact.scope_coverage_complete,
        "frv_routing_complete": artifact.routing_complete,
        "frv_membership_decision_complete": artifact.membership_decision_complete,
        "source_identity_complete": True,
        "xyz_recovery_complete": True,
        "hag_ready_complete": hag_ready_complete,
        "segmentation_input_complete": True,
        "v4_routing_complete": True,
        "measured_clearance_complete": False,
        "review_required": bool(review_count or unresolved_set.original_index.shape[0]),
        "failure_codes": (),
        "report_digest": "",
    }
    report = V4RecoveryCoverageReport.__new__(V4RecoveryCoverageReport)
    for name, value in values.items():
        object.__setattr__(report, name, value)
    if total != artifact.unique_in_scope_point_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED, field_name="route_counts"
        )
    object.__setattr__(report, "report_digest", report.computed_digest())
    return report


def _all_normal_ready(
    normal_indexes: list[int], payload_rows: Mapping[int, dict[str, np.ndarray]]
) -> bool:
    return all(
        int(payload_rows[index]["segmentation_hag_readiness"])
        == int(FrvSegmentationHagReadiness.READY)
        for index in normal_indexes
    )


# ---------------------------------------------------------------------------
# Coverage-only handoff
# ---------------------------------------------------------------------------


def _coverage_only_handoff(
    membership_result: FrvMembershipExecutionResult,
) -> V4InputHandoffResult:
    manifest = membership_result.coverage_manifest
    source = membership_result.source_identity
    records: list[V4UnrecoveredRegionRecord] = []
    for record in manifest.records:
        if record.review_required or record.failure_reason_flags:
            records.append(
                _unrecovered_region_record(membership_result, record)
            )
    report = _coverage_report(membership_result, records)
    handoff_digest = frv_v4_handoff_digest(
        source_content_identity_digest=source.source_content_identity_digest,
        frv_membership_artifact_digest=frv_digest(()),
        report_digest=report.report_digest,
        normal_set_digest=frv_digest(()),
        review_set_digest=frv_digest(()),
        unrecovered_regions=tuple(records),
    )
    return _make_handoff(
        validated_membership=None,
        recovered_vegetation=None,
        unresolved_measured_points=None,
        unrecovered_regions=tuple(records),
        coverage_report=report,
        output_reference=None,
        handoff_digest=handoff_digest,
    )


def _unrecovered_region_record(
    membership_result: FrvMembershipExecutionResult,
    record: Any,
) -> V4UnrecoveredRegionRecord:
    source = membership_result.source_identity
    payload = {
        "schema_version": V4_UNRECOVERED_REGION_SCHEMA_VERSION,
        "source_content_identity_digest": source.source_content_identity_digest,
        "frv_scope_coverage_manifest_digest": membership_result.coverage_manifest.manifest_digest,
        "canonical_inventory_key": record.canonical_inventory_key,
        "scope_id": record.scope_id,
        "frv_scope_union_id": None,
        "scope_union_geometry_digest": None,
        "failure_reason_flags": int(record.failure_reason_flags) or 1,
        "safety_relevance": "UNKNOWN",
        "review_required": True,
        "measured_clearance_complete": False,
        "diagnostic_reference": record.diagnostic_reference,
        "record_digest": "",
    }
    item = V4UnrecoveredRegionRecord.__new__(V4UnrecoveredRegionRecord)
    for name, value in payload.items():
        object.__setattr__(item, name, value)
    object.__setattr__(item, "record_digest", item.computed_digest())
    return item


def _coverage_report(
    membership_result: FrvMembershipExecutionResult,
    records: list[V4UnrecoveredRegionRecord],
) -> V4RecoveryCoverageReport:
    manifest = membership_result.coverage_manifest
    source = membership_result.source_identity
    values = {
        "schema_version": V4_RECOVERY_COVERAGE_SCHEMA_VERSION,
        "source_content_identity_digest": source.source_content_identity_digest,
        "source_identity_record_digest": source.source_identity_record_digest,
        "frv_scope_coverage_manifest_digest": manifest.manifest_digest,
        "frv_scope_union_id": manifest.scope_union_id or "0" * 64,
        "frv_upstream_inventory_digest": manifest.canonical_inventory_digest,
        "frv_membership_artifact_digest": "0" * 64,
        "recovered_index_digest": frv_v4_recovered_index_digest(
            source_content_identity_digest=source.source_content_identity_digest,
            original_index=np.asarray([], dtype=np.int64),
        ),
        "frv_canonical_inventory_count": manifest.canonical_inventory_count,
        "frv_required_inventory_count": manifest.required_inventory_count,
        "frv_approved_hard_rejection_count": manifest.approved_hard_rejection_count,
        "frv_scope_created_or_merged_count": manifest.scope_created_or_merged_count,
        "frv_coverage_blocked_count": manifest.coverage_blocked_count,
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
        "unrecovered_region_count": len(records),
        "frv_canonical_inventory_complete": manifest.canonical_inventory_complete,
        "frv_required_inventory_complete": manifest.required_inventory_complete,
        "frv_approved_hard_rejection_complete": manifest.approved_hard_rejection_complete,
        "frv_scope_coverage_complete": manifest.scope_coverage_complete,
        "frv_routing_complete": False,
        "frv_membership_decision_complete": False,
        "source_identity_complete": True,
        "xyz_recovery_complete": False,
        "hag_ready_complete": False,
        "segmentation_input_complete": False,
        "v4_routing_complete": True,
        "measured_clearance_complete": False,
        "review_required": True,
        "failure_codes": (),
        "report_digest": "",
    }
    report = V4RecoveryCoverageReport.__new__(V4RecoveryCoverageReport)
    for name, value in values.items():
        object.__setattr__(report, name, value)
    object.__setattr__(report, "report_digest", report.computed_digest())
    return report


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def _publish_ready_handoff(
    *,
    output_dir: Path,
    reference_dir: Path,
    view: V4ValidatedFrvMembershipView,
    coverage_report: V4RecoveryCoverageReport,
    recovered_set: V4RecoveredVegetationSet,
    unresolved_set: V4UnresolvedMeasuredPointSet,
    normal_indexes: list[int],
    review_indexes: list[int],
    normal_xyz: dict[int, np.ndarray],
    review_xyz: dict[int, np.ndarray],
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
    membership_artifact_digest: str,
    source_proof_digest: str,
    hag_proof_digest: str,
    geometry_proof_digest: str,
    wire_proof_digest: str,
) -> tuple[V4InputOutputReference, V4InputArtifactManifest]:
    normal_blocks = _encode_normal_blocks(
        normal_indexes, normal_xyz, payload_rows
    )
    review_blocks = _encode_review_blocks_from_sets(
        review_indexes, review_xyz, payload_rows, evidence_rows
    )
    normal_block_digests = tuple(
        hashlib.sha256(block).hexdigest() for block in normal_blocks
    )
    review_block_digests = tuple(
        hashlib.sha256(block).hexdigest() for block in review_blocks
    )
    normal_set_digest = frv_digest(
        {
            "digest_domain": "V4NRM1",
            "block_digests": list(normal_block_digests),
        }
    )
    review_set_digest = unresolved_set.set_digest
    handoff_digest = frv_v4_handoff_digest(
        source_content_identity_digest=view.source_identity.source_content_identity_digest,
        frv_membership_artifact_digest=membership_artifact_digest,
        report_digest=coverage_report.report_digest,
        normal_set_digest=normal_set_digest,
        review_set_digest=review_set_digest,
        unrecovered_regions=(),
    )
    _manifest_fields = {
        "schema_version": V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2,
        "source_content_identity_digest": view.source_identity.source_content_identity_digest,
        "source_identity_record_digest": view.source_identity.source_identity_record_digest,
        "origin_context": tuple(float(value) for value in view.source_identity.origin_context),
        "frv_membership_artifact_digest": membership_artifact_digest,
        "report_digest": coverage_report.report_digest,
        "source_proof_digest": source_proof_digest,
        "hag_proof_digest": hag_proof_digest,
        "geometry_proof_digest": geometry_proof_digest,
        "wire_proof_digest": wire_proof_digest,
        "normal_block_count": len(normal_blocks),
        "review_block_count": len(review_blocks),
        "normal_row_count": len(normal_indexes),
        "review_row_count": len(review_indexes),
        "normal_block_paths": tuple(
            f"normal/v4nrm-block-{ordinal:08d}.v4blk"
            for ordinal in range(len(normal_blocks))
        ),
        "review_block_paths": tuple(
            f"review/v4rev-block-{ordinal:08d}.v4blk"
            for ordinal in range(len(review_blocks))
        ),
        "normal_block_digests": normal_block_digests,
        "review_block_digests": review_block_digests,
        "normal_set_digest": normal_set_digest,
        "review_set_digest": review_set_digest,
        "handoff_digest": handoff_digest,
    }
    manifest = V4InputArtifactManifest(
        **_manifest_fields,
        manifest_digest=frv_digest(
            {
                "digest_domain": V4_INPUT_MANIFEST_DIGEST_DOMAIN,
                **_manifest_fields,
            }
        ),
    )

    def publish(staging: Path) -> None:
        (staging / "normal").mkdir()
        (staging / "review").mkdir()
        for ordinal, data in enumerate(normal_blocks):
            (staging / f"normal/v4nrm-block-{ordinal:08d}.v4blk").write_bytes(data)
        for ordinal, data in enumerate(review_blocks):
            (staging / f"review/v4rev-block-{ordinal:08d}.v4blk").write_bytes(data)
        write_canonical_json(staging / "validated_view.json", view.to_dict())
        write_canonical_json(staging / "coverage_report.json", coverage_report.to_dict())
        write_canonical_json(staging / "handoff_manifest.json", manifest.to_dict())

    publish_directory(scratch_dir=reference_dir, target_dir=output_dir, publish=publish)
    return (
        V4InputOutputReference(
            root_path=output_dir,
            validated_view_json_path=output_dir / "validated_view.json",
            coverage_json_path=output_dir / "coverage_report.json",
            normal_block_paths=tuple(
                output_dir / f"normal/v4nrm-block-{ordinal:08d}.v4blk"
                for ordinal in range(len(normal_blocks))
            ),
            review_block_paths=tuple(
                output_dir / f"review/v4rev-block-{ordinal:08d}.v4blk"
                for ordinal in range(len(review_blocks))
            ),
            handoff_manifest_path=output_dir / "handoff_manifest.json",
        ),
        manifest,
    )


def _encode_normal_blocks(
    normal_indexes: list[int],
    normal_xyz: dict[int, np.ndarray],
    payload_rows: Mapping[int, dict[str, np.ndarray]],
) -> list[bytes]:
    blocks: list[bytes] = []
    count = len(normal_indexes)
    start = 0
    while start < count:
        stop = min(start + _V4_BLOCK_CAPACITY, count)
        indexes = normal_indexes[start:stop]
        arrays: dict[str, np.ndarray] = {
            "original_index": np.asarray(indexes, dtype="<i8"),
            "xyz_local_m": np.asarray(
                [normal_xyz[index] for index in indexes], dtype="<f8"
            ),
            "hag_m": np.asarray(
                [float(payload_rows[index]["clamped_hag"]) for index in indexes], dtype="<f4"
            ),
            "membership_state": np.asarray(
                [int(payload_rows[index]["membership_state"]) for index in indexes], dtype="|u1"
            ),
            "hag_evidence_state": np.asarray(
                [int(payload_rows[index]["hag_evidence_state"]) for index in indexes], dtype="|u1"
            ),
            "segmentation_hag_readiness": np.asarray(
                [int(payload_rows[index]["segmentation_hag_readiness"]) for index in indexes],
                dtype="|u1",
            ),
            "candidate_reason_flags": np.asarray(
                [int(payload_rows[index]["reason_flags"]) for index in indexes], dtype="<u4"
            ),
        }
        for _name, array in arrays.items():
            array.flags.writeable = False
        blocks.append(
            encode_frv_block(
                magic=V4_NORMAL_BLOCK_MAGIC,
                artifact_id="v4-input-handoff",
                block_id=f"v4nrm-block-{len(blocks):08d}",
                payload_schema_digest="0" * 64,
                arrays=arrays,
                schema_order=tuple(name for name, _ in _NORMAL_FIELDS),
            )
        )
        start = stop
    if not blocks:
        arrays: dict[str, np.ndarray] = {
            "original_index": np.empty(0, dtype="<i8"),
            "xyz_local_m": np.empty((0, 3), dtype="<f8"),
            "hag_m": np.empty(0, dtype="<f4"),
            "membership_state": np.empty(0, dtype="|u1"),
            "hag_evidence_state": np.empty(0, dtype="|u1"),
            "segmentation_hag_readiness": np.empty(0, dtype="|u1"),
            "candidate_reason_flags": np.empty(0, dtype="<u4"),
        }
        blocks.append(
            encode_frv_block(
                magic=V4_NORMAL_BLOCK_MAGIC,
                artifact_id="v4-input-handoff",
                block_id="v4nrm-block-00000000",
                payload_schema_digest="0" * 64,
                arrays=arrays,
                schema_order=tuple(name for name, _ in _NORMAL_FIELDS),
            )
        )
    return blocks


def _encode_review_blocks_from_sets(
    review_indexes: list[int],
    review_xyz: dict[int, np.ndarray],
    payload_rows: Mapping[int, dict[str, np.ndarray]],
    evidence_rows: Mapping[int, dict[str, np.ndarray]],
) -> list[bytes]:
    count = len(review_indexes)
    xyz = np.zeros((count, 3), dtype=np.float64)
    for position, index in enumerate(review_indexes):
        xyz[position, :] = review_xyz[index]
    return _encode_review_blocks(review_indexes, xyz, payload_rows, evidence_rows)


def _set_block_digests(paths: tuple[Path, ...]) -> tuple[str, ...]:
    return tuple(hashlib.sha256(path.read_bytes()).hexdigest() for path in paths)


# ---------------------------------------------------------------------------
# Public v2 handoff loader
# ---------------------------------------------------------------------------


def _decode_v4_block_file(
    path: Path,
    *,
    magic: bytes,
    schema_order: tuple[str, ...],
    dtypes: Mapping[str, np.dtype],
) -> dict[str, np.ndarray]:
    row_shapes: dict[str, tuple[int, ...]] = {
        name: () for name in schema_order
    }
    if "xyz_local_m" in row_shapes:
        row_shapes["xyz_local_m"] = (3,)
    _header, arrays = decode_frv_block(
        path.read_bytes(),
        magic=magic,
        schema_order=schema_order,
        dtypes=dtypes,
        row_shapes=row_shapes,
    )
    return arrays


def _normal_block_dtypes() -> dict[str, np.dtype]:
    return {name: np.dtype(dtype) for name, dtype in _NORMAL_FIELDS}


def _review_block_dtypes() -> dict[str, np.dtype]:
    return {name: np.dtype(dtype) for name, dtype in _REVIEW_FIELDS}


def _load_ready_blocks_from_manifest(
    root: Path,
    manifest: V4InputArtifactManifest,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    normal_paths = [root / path for path in manifest.normal_block_paths]
    review_paths = [root / path for path in manifest.review_block_paths]
    for path in normal_paths + review_paths:
        if not path.is_file():
            raise V4InputLoadError(
                code=V4InputErrorCode.OUTPUT_PUBLICATION_FAILED,
                field_name="block",
            )
    for expected, path in zip(manifest.normal_block_digests, normal_paths, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise V4InputLoadError(
                code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="normal_block"
            )
    for expected, path in zip(manifest.review_block_digests, review_paths, strict=True):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise V4InputLoadError(
                code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="review_block"
            )
    normal_arrays = [
        _decode_v4_block_file(
            path, magic=V4_NORMAL_BLOCK_MAGIC, schema_order=_normal_dtype_order(),
            dtypes=_normal_block_dtypes(),
        )
        for path in normal_paths
    ]
    review_arrays = [
        _decode_v4_block_file(
            path, magic=V4_REVIEW_BLOCK_MAGIC, schema_order=_review_dtype_order(),
            dtypes=_review_block_dtypes(),
        )
        for path in review_paths
    ]
    return _concatenate_block_arrays(normal_arrays), _concatenate_block_arrays(
        review_arrays
    )


def _normal_dtype_order() -> tuple[str, ...]:
    return tuple(name for name, _dtype in _NORMAL_FIELDS)


def _review_dtype_order() -> tuple[str, ...]:
    return tuple(name for name, _dtype in _REVIEW_FIELDS)


def _concatenate_block_arrays(blocks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not blocks:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="block"
        )
    names = tuple(blocks[0])
    result: dict[str, np.ndarray] = {}
    for name in names:
        parts = [block[name] for block in blocks]
        result[name] = np.concatenate(parts, axis=0)
        result[name].setflags(write=False)
    return result


def _validate_source_row_identity(
    source_identity: SourceIdentity,
    normal: Mapping[str, np.ndarray],
    review: Mapping[str, np.ndarray],
) -> None:
    """Bind every persisted V4 row identity back to the trusted source bytes.

    The v2 block/set/report/manifest digests are an integrity chain, not an
    authenticity primitive: an internally rehashed block must still fail when
    its ``original_index`` no longer names the persisted measured XYZ.  Replay
    the frozen LAS source once, verify its content hash/count, and compare only
    the requested normal/review rows under the authoritative origin context.
    """

    source_path = Path(source_identity.resolved_access_path)
    if not source_path.is_file() or source_path.suffix.lower() != ".las":
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_replay",
        )
    hasher = hashlib.sha256()
    try:
        with source_path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    except OSError as exc:
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_replay",
        ) from exc
    if hasher.hexdigest() != source_identity.source_file_sha256:
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_file_sha256",
        )

    normal_indexes = np.asarray(normal["original_index"], dtype=np.int64)
    review_indexes = np.asarray(review["original_index"], dtype=np.int64)
    requested = np.concatenate((normal_indexes, review_indexes))
    if requested.size and (
        int(requested.min()) < 0
        or int(requested.max()) >= source_identity.source_point_count
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_INVALID,
            field_name="original_index",
        )
    expected_by_index = {
        int(index): np.asarray(xyz, dtype=np.float64)
        for indexes, xyz_rows in (
            (normal_indexes, normal["xyz_local_m"]),
            (review_indexes, review["xyz_local_m"]),
        )
        for index, xyz in zip(indexes, np.asarray(xyz_rows), strict=True)
    }
    origin = np.asarray(source_identity.origin_context, dtype=np.float64)
    seen: set[int] = set()
    sorted_targets = sorted(expected_by_index)
    target_cursor = 0
    offset = 0
    try:
        with laspy.open(source_path) as reader:
            if int(reader.header.point_count) != source_identity.source_point_count:
                raise V4InputLoadError(
                    code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
                    field_name="source_point_count",
                )
            for points in reader.chunk_iterator(250_000):
                stop = offset + len(points)
                while (
                    target_cursor < len(sorted_targets)
                    and sorted_targets[target_cursor] < stop
                ):
                    index = sorted_targets[target_cursor]
                    target_cursor += 1
                    if index < offset:
                        continue
                    row = index - offset
                    actual_local = np.asarray(
                        [points.x[row], points.y[row], points.z[row]],
                        dtype=np.float64,
                    ) - origin
                    if not np.allclose(
                        actual_local,
                        expected_by_index[index],
                        rtol=0.0,
                        atol=1e-9,
                    ):
                        raise V4InputLoadError(
                            code=V4InputErrorCode.ORIGINAL_INDEX_INVALID,
                            field_name="source_row_identity",
                        )
                    seen.add(index)
                offset = stop
    except V4InputLoadError:
        raise
    except (OSError, ValueError, laspy.errors.LaspyException) as exc:
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_replay",
        ) from exc
    if seen != set(expected_by_index):
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_INVALID,
            field_name="source_row_identity",
        )


def _check_normal_routes(normal: dict[str, np.ndarray]) -> None:
    membership = np.asarray(normal["membership_state"], dtype=np.uint8)
    readiness = np.asarray(normal["segmentation_hag_readiness"], dtype=np.uint8)
    admitted = (
        (membership == int(FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE))
        | (membership == int(FrvMembershipState.MANUAL_INCLUDE))
    )
    if not np.all(admitted):
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="normal_membership",
        )
    if not np.all(readiness == int(FrvSegmentationHagReadiness.READY)):
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="normal_readiness",
        )


def _check_review_routes(review: dict[str, np.ndarray]) -> None:
    routes = np.asarray(review["primary_route"], dtype=np.uint8)
    if not np.all(routes == int(V4PrimaryRoute.MEASURED_POINT_REVIEW)):
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="review_route",
        )


def _recovered_set_from_arrays(
    source_identity: SourceIdentity,
    normal: dict[str, np.ndarray],
    frv_membership_artifact_digest: str,
) -> V4RecoveredVegetationSet:
    indexes = np.asarray(normal["original_index"], dtype=np.int64)
    if indexes.size == 0:
        indexes = indexes.astype("<i8")
    set_obj = V4RecoveredVegetationSet.__new__(V4RecoveredVegetationSet)
    object.__setattr__(set_obj, "source_identity", source_identity)
    object.__setattr__(set_obj, "original_index", _ro(indexes.astype("<i8")))
    object.__setattr__(set_obj, "xyz_local_m", _ro(np.asarray(normal["xyz_local_m"]).copy()))
    object.__setattr__(set_obj, "hag_m", _ro(np.asarray(normal["hag_m"]).copy()))
    object.__setattr__(
        set_obj, "membership_state", _ro(np.asarray(normal["membership_state"]).copy())
    )
    object.__setattr__(
        set_obj,
        "hag_evidence_state",
        _ro(np.asarray(normal["hag_evidence_state"]).copy()),
    )
    object.__setattr__(
        set_obj,
        "segmentation_hag_readiness",
        _ro(np.asarray(normal["segmentation_hag_readiness"]).copy()),
    )
    object.__setattr__(
        set_obj, "candidate_reason_flags", _ro(np.asarray(normal["candidate_reason_flags"]).copy())
    )
    object.__setattr__(set_obj, "candidate_confidence", None)
    object.__setattr__(
        set_obj, "frv_membership_artifact_digest", frv_membership_artifact_digest
    )
    object.__setattr__(
        set_obj,
        "recovered_index_digest",
        frv_v4_recovered_index_digest(
            source_content_identity_digest=source_identity.source_content_identity_digest,
            original_index=indexes,
        ),
    )
    return set_obj


def _unresolved_set_from_arrays(
    source_identity: SourceIdentity,
    review: dict[str, np.ndarray],
    frv_membership_artifact_digest: str,
    source_proof_digest: str,
    hag_proof_digest: str,
    geometry_proof_digest: str,
    wire_proof_digest: str,
    review_set_digest: str,
) -> V4UnresolvedMeasuredPointSet:
    set_obj = V4UnresolvedMeasuredPointSet.__new__(V4UnresolvedMeasuredPointSet)
    object.__setattr__(set_obj, "source_identity", source_identity)
    object.__setattr__(
        set_obj, "original_index", _ro(np.asarray(review["original_index"]).copy())
    )
    object.__setattr__(set_obj, "xyz_local_m", _ro(np.asarray(review["xyz_local_m"]).copy()))
    object.__setattr__(
        set_obj, "membership_state", _ro(np.asarray(review["membership_state"]).copy())
    )
    object.__setattr__(
        set_obj, "hag_evidence_state", _ro(np.asarray(review["hag_evidence_state"]).copy())
    )
    object.__setattr__(
        set_obj,
        "segmentation_hag_readiness",
        _ro(np.asarray(review["segmentation_hag_readiness"]).copy()),
    )
    object.__setattr__(set_obj, "raw_hag_m", _ro(np.asarray(review["raw_hag_m"]).copy()))
    object.__setattr__(
        set_obj, "clamped_hag_m", _ro(np.asarray(review["clamped_hag_m"]).copy())
    )
    object.__setattr__(set_obj, "reason_flags", _ro(np.asarray(review["reason_flags"]).copy()))
    object.__setattr__(set_obj, "quality_flags", _ro(np.asarray(review["quality_flags"]).copy()))
    object.__setattr__(
        set_obj, "wire_provider_state", _ro(np.asarray(review["wire_provider_state"]).copy())
    )
    object.__setattr__(
        set_obj, "wire_relation_state", _ro(np.asarray(review["wire_relation_state"]).copy())
    )
    object.__setattr__(
        set_obj, "nearest_wire_id", _ro(np.asarray(review["nearest_wire_id"]).copy())
    )
    object.__setattr__(
        set_obj, "centerline_distance_m", _ro(np.asarray(review["centerline_distance_m"]).copy())
    )
    object.__setattr__(
        set_obj, "primary_route", _ro(np.asarray(review["primary_route"]).copy())
    )
    object.__setattr__(
        set_obj, "frv_membership_artifact_digest", frv_membership_artifact_digest
    )
    object.__setattr__(set_obj, "source_proof_digest", source_proof_digest)
    object.__setattr__(set_obj, "hag_proof_digest", hag_proof_digest)
    object.__setattr__(set_obj, "geometry_proof_digest", geometry_proof_digest)
    object.__setattr__(set_obj, "wire_proof_digest", wire_proof_digest)
    object.__setattr__(set_obj, "set_digest", review_set_digest)
    return set_obj


def load_v4_input_handoff(root_path: Path) -> V4InputHandoffResult:
    """Load and re-validate a formal digest-bound v2 V4 input handoff root.

    This is the formal segmentation entry point: it accepts only v2 manifests
    with a persisted block digest inventory, set digests and the final handoff
    digest.  v1 roots (which lack the persisted inventory) fail closed here and
    remain available only through the read-only FRV-VIS-A1 loader.
    """

    if not isinstance(root_path, Path) or not root_path.is_dir():
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="root"
        )
    from core.contracts.v4_inputs import (
        V4InputArtifactManifest,
        validate_v4_input_artifact_manifest,
    )

    manifest_path = root_path / "handoff_manifest.json"
    if not manifest_path.is_file():
        raise V4InputLoadError(
            code=V4InputErrorCode.OUTPUT_PUBLICATION_FAILED,
            field_name="handoff_manifest.json",
        )
    try:
        payload = json.loads(manifest_path.read_text("utf-8"))
        if payload.get("schema_version") != V4_HANDOFF_MANIFEST_SCHEMA_VERSION_V2:
            raise V4InputLoadError(
                code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="schema_version"
            )
        manifest = V4InputArtifactManifest.from_dict(payload)
        validate_v4_input_artifact_manifest(manifest)
    except V4InputLoadError:
        raise
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="handoff_manifest.json"
        ) from exc

    view_path = root_path / "validated_view.json"
    report_path = root_path / "coverage_report.json"
    if not view_path.is_file() or not report_path.is_file():
        raise V4InputLoadError(
            code=V4InputErrorCode.OUTPUT_PUBLICATION_FAILED,
            field_name="ready_files",
        )
    try:
        view_payload = json.loads(view_path.read_text("utf-8"))
        view_payload["membership_block_references"] = ()
        view_payload["decision_evidence_block_references"] = ()
        view = V4ValidatedFrvMembershipView.from_dict(view_payload)
        coverage_report = V4RecoveryCoverageReport.from_dict(
            json.loads(report_path.read_text("utf-8"))
        )
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="ready_files"
        ) from exc

    # Run the full contract validators on the reconstructed objects rather than
    # trusting them merely because their stored digests are self-consistent.
    try:
        validate_frv_validated_membership_view(view)
        validate_frv_recovery_coverage_report(coverage_report)
    except (ValueError, TypeError) as exc:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="ready_files"
        ) from exc
    # The formal segmentation loader requires a fully READY handoff: every
    # frozen completeness flag must be true.  A false flag here is a hard fail,
    # not a review or a gap.
    for name in (
        "frv_canonical_inventory_complete",
        "frv_required_inventory_complete",
        "frv_approved_hard_rejection_complete",
        "frv_scope_coverage_complete",
        "frv_routing_complete",
        "frv_membership_decision_complete",
        "source_identity_complete",
        "xyz_recovery_complete",
        "hag_ready_complete",
        "segmentation_input_complete",
        "v4_routing_complete",
    ):
        if not bool(getattr(coverage_report, name)):
            raise V4InputLoadError(
                code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
                field_name=name,
            )
    for name in (
        "frv_canonical_inventory_complete",
        "frv_required_inventory_complete",
        "frv_approved_hard_rejection_complete",
        "frv_scope_coverage_complete",
        "frv_routing_complete",
        "frv_membership_decision_complete",
    ):
        if not bool(getattr(view, name)):
            raise V4InputLoadError(
                code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
                field_name=name,
            )
    if coverage_report.measured_clearance_complete is not False:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="measured_clearance_complete",
        )

    # The view, its embedded FRV artifact, and the V4 coverage report are three
    # independently digested records.  Bind every shared count/completeness
    # field here so a rehashed outer bundle cannot substitute a self-consistent
    # report that disagrees with the upstream semantic artifact.
    artifact = view.membership_artifact
    shared_fields = (
        "frv_canonical_inventory_count",
        "frv_required_inventory_count",
        "frv_approved_hard_rejection_count",
        "frv_scope_created_or_merged_count",
        "frv_coverage_blocked_count",
        "frv_unique_in_scope_point_count",
        "frv_evaluated_point_count",
        "frv_unasserted_in_scope_count",
        "frv_verified_vegetation_candidate_count",
        "frv_excluded_non_candidate_count",
        "frv_review_required_membership_count",
        "frv_manual_include_count",
        "frv_manual_exclude_count",
        "frv_canonical_inventory_complete",
        "frv_required_inventory_complete",
        "frv_approved_hard_rejection_complete",
        "frv_scope_coverage_complete",
        "frv_routing_complete",
        "frv_membership_decision_complete",
    )
    for name in shared_fields:
        if getattr(view, name) != getattr(coverage_report, name):
            raise V4InputLoadError(
                code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
                field_name=name,
            )
    artifact_field_by_view_field = {
        "frv_canonical_inventory_count": "canonical_inventory_count",
        "frv_required_inventory_count": "required_inventory_count",
        "frv_approved_hard_rejection_count": "approved_hard_rejection_count",
        "frv_scope_created_or_merged_count": "scope_created_or_merged_count",
        "frv_coverage_blocked_count": "coverage_blocked_count",
        "frv_unique_in_scope_point_count": "unique_in_scope_point_count",
        "frv_evaluated_point_count": "evaluated_point_count",
        "frv_unasserted_in_scope_count": "unasserted_in_scope_count",
        "frv_verified_vegetation_candidate_count": (
            "verified_vegetation_candidate_count"
        ),
        "frv_excluded_non_candidate_count": "excluded_non_candidate_count",
        "frv_review_required_membership_count": "review_required_membership_count",
        "frv_manual_include_count": "manual_include_count",
        "frv_manual_exclude_count": "manual_exclude_count",
        "frv_canonical_inventory_complete": "canonical_inventory_complete",
        "frv_required_inventory_complete": "required_inventory_complete",
        "frv_approved_hard_rejection_complete": "approved_hard_rejection_complete",
        "frv_scope_coverage_complete": "scope_coverage_complete",
        "frv_routing_complete": "routing_complete",
        "frv_membership_decision_complete": "membership_decision_complete",
    }
    for view_field, artifact_field in artifact_field_by_view_field.items():
        if getattr(view, view_field) != getattr(artifact, artifact_field):
            raise V4InputLoadError(
                code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
                field_name=view_field,
            )
    if (
        view.coverage_manifest.manifest_digest
        != coverage_report.frv_scope_coverage_manifest_digest
        or artifact.scope_coverage_manifest_digest
        != coverage_report.frv_scope_coverage_manifest_digest
        or artifact.manifest_digest != coverage_report.frv_membership_artifact_digest
        or view.artifact_binding.membership_artifact_manifest_digest
        != artifact.manifest_digest
        or view.coverage_manifest.canonical_inventory_digest
        != coverage_report.frv_upstream_inventory_digest
        or artifact.scope_union_id != coverage_report.frv_scope_union_id
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID,
            field_name="frv_artifact_binding",
        )

    if (
        manifest.source_content_identity_digest
        != view.source_identity.source_content_identity_digest
        or manifest.source_content_identity_digest
        != coverage_report.source_content_identity_digest
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_content_identity_digest",
        )
    if (
        manifest.source_identity_record_digest
        != view.source_identity.source_identity_record_digest
        or manifest.source_identity_record_digest
        != coverage_report.source_identity_record_digest
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED,
            field_name="source_identity_record_digest",
        )
    if manifest.origin_context != tuple(view.source_identity.origin_context):
        raise V4InputLoadError(
            code=V4InputErrorCode.SOURCE_IDENTITY_FAILED, field_name="origin_context"
        )
    if manifest.report_digest != coverage_report.report_digest:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="report_digest"
        )
    if (
        manifest.frv_membership_artifact_digest
        != coverage_report.frv_membership_artifact_digest
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID,
            field_name="frv_membership_artifact_digest",
        )

    normal, review = _load_ready_blocks_from_manifest(root_path, manifest)
    normal_row_count = int(np.asarray(normal["original_index"]).shape[0])
    review_row_count = int(np.asarray(review["original_index"]).shape[0])
    if normal_row_count != manifest.normal_row_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="normal_row_count",
        )
    if review_row_count != manifest.review_row_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="review_row_count",
        )
    if normal_row_count != coverage_report.normal_segmentation_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="normal_segmentation_count",
        )
    if review_row_count != coverage_report.measured_point_review_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="measured_point_review_count",
        )
    _check_normal_routes(normal)
    _check_review_routes(review)
    _validate_source_row_identity(view.source_identity, normal, review)
    persisted_membership = np.concatenate(
        (
            np.asarray(normal["membership_state"], dtype=np.uint8),
            np.asarray(review["membership_state"], dtype=np.uint8),
        )
    )
    state_count_fields = (
        (
            FrvMembershipState.VERIFIED_VEGETATION_CANDIDATE,
            "frv_verified_vegetation_candidate_count",
        ),
        (
            FrvMembershipState.REVIEW_REQUIRED_MEMBERSHIP,
            "frv_review_required_membership_count",
        ),
        (FrvMembershipState.MANUAL_INCLUDE, "frv_manual_include_count"),
    )
    for state, field_name in state_count_fields:
        observed_count = int(np.count_nonzero(persisted_membership == int(state)))
        if observed_count != getattr(coverage_report, field_name):
            raise V4InputLoadError(
                code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
                field_name=field_name,
            )
    expected_excluded_count = (
        coverage_report.frv_excluded_non_candidate_count
        + coverage_report.frv_manual_exclude_count
    )
    if coverage_report.excluded_audit_count != expected_excluded_count:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="excluded_audit_count",
        )
    if normal_row_count + review_row_count + expected_excluded_count != (
        coverage_report.frv_unique_in_scope_point_count
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="frv_unique_in_scope_point_count",
        )

    # The canonical publisher writes every normal block in ascending
    # original_index order; the persisted block order itself is part of the
    # handoff identity.  An attacker reordering rows inside a block would
    # produce a recomputed recovered-index digest that disagrees with the
    # coverage report authority.
    normal_indexes = np.asarray(normal["original_index"], dtype=np.int64)
    if normal_indexes.size > 1 and not (np.diff(normal_indexes) > 0).all():
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="normal_index_order",
        )
    if int(normal_indexes.shape[0]) != int(np.unique(normal_indexes).shape[0]):
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_DUPLICATE, field_name="normal_index"
        )
    if int(review["original_index"].shape[0]) != int(
        np.unique(np.asarray(review["original_index"], dtype=np.int64)).shape[0]
    ):
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_DUPLICATE, field_name="review_index"
        )
    overlapping = np.intersect1d(
        normal_indexes, np.asarray(review["original_index"], dtype=np.int64)
    )
    if overlapping.size:
        raise V4InputLoadError(
            code=V4InputErrorCode.ROUTING_CONSERVATION_FAILED,
            field_name="route_overlap",
        )

    normal_set_digest = frv_digest(
        {
            "digest_domain": "V4NRM1",
            "block_digests": list(manifest.normal_block_digests),
        }
    )
    review_set_digest = frv_digest(
        {
            "digest_domain": "V4REV1",
            "block_digests": list(manifest.review_block_digests),
        }
    )
    if normal_set_digest != manifest.normal_set_digest:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="normal_set_digest"
        )
    if review_set_digest != manifest.review_set_digest:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="review_set_digest"
        )
    handoff_digest = frv_v4_handoff_digest(
        source_content_identity_digest=manifest.source_content_identity_digest,
        frv_membership_artifact_digest=manifest.frv_membership_artifact_digest,
        report_digest=manifest.report_digest,
        normal_set_digest=normal_set_digest,
        review_set_digest=review_set_digest,
        unrecovered_regions=(),
    )
    if handoff_digest != manifest.handoff_digest:
        raise V4InputLoadError(
            code=V4InputErrorCode.FRV_ARTIFACT_INVALID, field_name="handoff_digest"
        )

    # Independently reconstructed recovered-index authority: recompute the
    # V4RECIDX1 digest over the actual normal block bytes and bind it to the
    # coverage report.  A self-consistent manifest forgery (block bytes +
    # manifest values + recomputed manifest digest) cannot pass unless it also
    # forges the coverage report and the handoff identity chain.
    recovered_index_digest = frv_v4_recovered_index_digest(
        source_content_identity_digest=view.source_identity.source_content_identity_digest,
        original_index=normal_indexes,
    )
    if recovered_index_digest != coverage_report.recovered_index_digest:
        raise V4InputLoadError(
            code=V4InputErrorCode.ORIGINAL_INDEX_INVALID,
            field_name="recovered_index_digest",
        )
    # Reconstruct the formal sets through their real constructors so every
    # full contract invariant (dtype/shape/finite/read-only) runs; do not
    # build a ``__new__`` object that is then trusted without validation.
    recovered_set = V4RecoveredVegetationSet(
        source_identity=view.source_identity,
        original_index=_ro(np.asarray(normal_indexes, dtype="<i8")),
        xyz_local_m=_ro(np.asarray(normal["xyz_local_m"]).copy()),
        hag_m=_ro(np.asarray(normal["hag_m"]).copy()),
        membership_state=_ro(np.asarray(normal["membership_state"]).copy()),
        hag_evidence_state=_ro(np.asarray(normal["hag_evidence_state"]).copy()),
        segmentation_hag_readiness=_ro(
            np.asarray(normal["segmentation_hag_readiness"]).copy()
        ),
        candidate_reason_flags=_ro(np.asarray(normal["candidate_reason_flags"]).copy()),
        candidate_confidence=None,
        frv_membership_artifact_digest=manifest.frv_membership_artifact_digest,
        recovered_index_digest=recovered_index_digest,
    )
    unresolved_set = V4UnresolvedMeasuredPointSet(
        source_identity=view.source_identity,
        original_index=_ro(np.asarray(review["original_index"]).copy()),
        xyz_local_m=_ro(np.asarray(review["xyz_local_m"]).copy()),
        membership_state=_ro(np.asarray(review["membership_state"]).copy()),
        hag_evidence_state=_ro(np.asarray(review["hag_evidence_state"]).copy()),
        segmentation_hag_readiness=_ro(
            np.asarray(review["segmentation_hag_readiness"]).copy()
        ),
        raw_hag_m=_ro(np.asarray(review["raw_hag_m"]).copy()),
        clamped_hag_m=_ro(np.asarray(review["clamped_hag_m"]).copy()),
        reason_flags=_ro(np.asarray(review["reason_flags"]).copy()),
        quality_flags=_ro(np.asarray(review["quality_flags"]).copy()),
        wire_provider_state=_ro(np.asarray(review["wire_provider_state"]).copy()),
        wire_relation_state=_ro(np.asarray(review["wire_relation_state"]).copy()),
        nearest_wire_id=_ro(np.asarray(review["nearest_wire_id"]).copy()),
        centerline_distance_m=_ro(np.asarray(review["centerline_distance_m"]).copy()),
        primary_route=_ro(np.asarray(review["primary_route"]).copy()),
        frv_membership_artifact_digest=manifest.frv_membership_artifact_digest,
        source_proof_digest=manifest.source_proof_digest,
        hag_proof_digest=manifest.hag_proof_digest,
        geometry_proof_digest=manifest.geometry_proof_digest,
        wire_proof_digest=manifest.wire_proof_digest,
        set_digest=review_set_digest,
    )
    reference = V4InputOutputReference(
        root_path=root_path,
        validated_view_json_path=view_path,
        coverage_json_path=report_path,
        normal_block_paths=tuple(
            root_path / path for path in manifest.normal_block_paths
        ),
        review_block_paths=tuple(
            root_path / path for path in manifest.review_block_paths
        ),
        handoff_manifest_path=manifest_path,
    )
    return _make_handoff(
        validated_membership=view,
        recovered_vegetation=recovered_set,
        unresolved_measured_points=unresolved_set,
        unrecovered_regions=(),
        coverage_report=coverage_report,
        output_reference=reference,
        handoff_digest=manifest.handoff_digest,
    )


__all__ = [
    "build_v4_input_handoff",
    "load_v4_input_handoff",
    "V4_NORMAL_BLOCK_MAGIC",
    "V4_REVIEW_BLOCK_MAGIC",
]
