"""FRV-A2 P-HAG runtime provider (PLAN-FROZEN).

P-HAG consumes a finalized P-SOURCE result and runs exactly two independent
``load_frv_source`` sessions (Pass 1 builds the sparse classification-2 ground
grid; Pass 2 emits one HAG chunk per source chunk and accumulates the HAG
point-content digest).  No direct LAS access, no third pass, no full source
materialization.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator
from types import TracebackType
from typing import Any, Literal

import numpy as np
from core.contracts.frv import (
    FrvConfiguration,
    FrvErrorCode,
    FrvHagEvidenceState,
    FrvQualityFlag,
    FrvReasonFlag,
    frv_digest,
    validate_configuration,
)
from core.contracts.frv_a2 import (
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
    _derive_artifact_reference_id,
)
from core.contracts.frv_hag import (
    _FRV_HAG_FINALIZED_CAPABILITY,
    FRV_HAG_ALGORITHM_VERSION,
    FRV_HAG_PROOF_SCHEMA_VERSION,
    FRV_HAG_PROVIDER_INTERFACE_VERSION,
    FRV_HAG_PROVIDER_KEY,
    FRV_HAG_SOURCE_PROOF_DIGEST_DOMAIN,
    FrvHagChunk,
    FrvHagLoadError,
    FrvHagLoadSession,
    FrvHagProviderConfig,
    FrvHagProviderProof,
    FrvHagProviderResult,
    _create_finalized_frv_hag_provider_result,
    frv_hag_ground_cell_inventory_digest,
    validate_frv_hag_provider_config,
)
from core.contracts.frv_source import (
    _FRV_SOURCE_FINALIZED_CAPABILITY,
    FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
    FRV_SOURCE_PROVIDER_KEY,
    FrvSourceCanonicalContentPolicy,
    FrvSourceCoordinateConversionPolicy,
    FrvSourceInvalidPointPolicy,
    FrvSourceModel,
    FrvSourcePassThroughDimension,
    FrvSourceProviderConfig,
    FrvSourceProviderProof,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
    derive_frv_source_provider_config_digest,
    to_frv_a2_provider_binding_evidence,
    validate_finalized_frv_source_provider_result,
    validate_frv_source_provider_reference,
)

from algorithms.io.source_provider import load_frv_source

# Private session-state markers (never exposed).
_CREATED = "CREATED"
_ITERATING = "ITERATING"
_STREAM_COMPLETE = "STREAM_COMPLETE"
_FINALIZED = "FINALIZED"
_ABORTED = "ABORTED"
_CLOSED = "CLOSED"

# Frozen plan section 6 mapping keys -> failure-record tuple.  Only existing A2
# enums/flags are used.
_HAG_FAILURE_MAP: dict[
    str, tuple[FrvA2FailurePhase, str, FrvErrorCode, FrvReasonFlag, FrvQualityFlag]
] = {
    "SOURCE_RESULT_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "source_result",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "SOURCE_REFERENCE_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "reference",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "SOURCE_BINDING_MISMATCH": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "source_result",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "CLASSIFICATION_MISSING": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "classification",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "CLASSIFICATION_DTYPE": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "classification",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "NO_GROUND": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "ground_points",
        FrvErrorCode.FRV_HAG_MISSING,
        FrvReasonFlag.PROVIDER_UNCERTAINTY | FrvReasonFlag.HAG_MISSING,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "PROVIDER_CONFIG_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "provider_config",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "COORDINATE_FRAME_INVALID": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "coordinate_frame",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "UNITS_INVALID": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "units",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "TRANSFORM_INVALID": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "transform_declaration",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "SOURCE_PASS_INCOMPLETE": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "source_pass",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "GRID_BUILD": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "ground_grid",
        FrvErrorCode.FRV_HAG_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "NONFINITE_XYZ": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "xyz",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "NONFINITE_GROUND": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "ground_z",
        FrvErrorCode.FRV_HAG_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "NONFINITE_INTERPOLATION": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "interpolation",
        FrvErrorCode.FRV_HAG_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "INDEX_LOST": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "original_index",
        FrvErrorCode.FRV_ORIGINAL_INDEX_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.ORIGINAL_INDEX_INVALID,
    ),
    "INDEX_DUPLICATE": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "original_index",
        FrvErrorCode.FRV_ORIGINAL_INDEX_DUPLICATE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.ORIGINAL_INDEX_DUPLICATE,
    ),
    "INDEX_OUT_OF_RANGE": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "original_index",
        FrvErrorCode.FRV_ORIGINAL_INDEX_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.ORIGINAL_INDEX_INVALID,
    ),
    "POINT_COUNT_MISMATCH": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "point_count",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "DIGEST_MISMATCH": (
        FrvA2FailurePhase.CROSS_PROVIDER,
        "digest",
        FrvErrorCode.FRV_ARTIFACT_DIGEST_MISMATCH,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.ARTIFACT_DIGEST_MISMATCH,
    ),
    "REPEATED_ITERATOR": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "session_state",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "PREMATURE_FINALIZE": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "session_state",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "REPEATED_FINALIZE": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "session_state",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "CLEANUP_FAILURE": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "cleanup",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
}


def _build_hag_error(key: str) -> FrvHagLoadError:
    phase, field_name, error_code, reason_flags, quality_flag = _HAG_FAILURE_MAP[key]
    detail = (
        f"{phase.value}:{FrvA2ProviderRole.HAG.value}:{FRV_HAG_PROVIDER_KEY}:"
        f"{field_name}:{error_code.value}:{int(reason_flags)}"
    )
    record = FrvA2FailureRecord(
        phase,
        FrvA2ProviderRole.HAG,
        FRV_HAG_PROVIDER_KEY,
        field_name,
        error_code,
        int(reason_flags),
        detail,
    )
    return FrvHagLoadError(failure_record=record, quality_flags=int(quality_flag))


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _reconstruct_frv_source_provider_config(
    source_proof: FrvSourceProviderProof,
) -> FrvSourceProviderConfig:
    """Reconstruct the exact P-SOURCE config from the frozen proof fields."""

    pass_through_dimensions = tuple(source_proof.pass_through_dimensions)
    provider_config_digest = derive_frv_source_provider_config_digest(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=pass_through_dimensions,
    )
    if provider_config_digest != source_proof.provider_config_digest:
        raise _build_hag_error("PROVIDER_CONFIG_INVALID")
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=pass_through_dimensions,
        provider_config_digest=provider_config_digest,
    )


class _FrvHagLoadSession:
    """Single-use two-pass P-HAG session (private; not exported)."""

    def __init__(
        self,
        reference: FrvSourceProviderReference,
        source_result: FrvSourceProviderResult,
        frv_configuration: FrvConfiguration,
        provider_config: FrvHagProviderConfig,
        source_chunk_size: int,
    ) -> None:
        self._reference = reference
        self._source_result = source_result
        self._frv_configuration = frv_configuration
        self._provider_config = provider_config
        self._source_chunk_size = source_chunk_size
        self._state = _CREATED
        self._had_failure = False
        self._result: FrvHagProviderResult | None = None
        self._result2: FrvSourceProviderResult | None = None
        self._grid: dict[tuple[int, int], tuple[float, int]] = {}
        self._ground_point_count = 0
        self._source_point_count = int(source_result.proof.source_point_count)
        self._expected_source_id = str(source_result.source_identity.source_id)
        self._pass1_source_id: str | None = None
        self._hag_pcd1: Any = None
        self._expected_hag_index = 0
        self._supported_count = 0
        self._unsupported_count = 0
        self._valid_positive_count = 0
        self._valid_near_zero_count = 0
        self._within_negative_count = 0
        self._missing_count = 0
        self._invalid_count = 0
        self._active_source_sessions: list[Any] = []
        self._pre_pass_validate()

    # ------------------------------------------------------------------
    # Pre-pass validation (frozen first-failure order)
    # ------------------------------------------------------------------

    def _pre_pass_validate(self) -> None:
        if type(self._source_result) is not FrvSourceProviderResult:
            raise _build_hag_error("SOURCE_RESULT_INVALID")
        if (
            getattr(self._source_result, "_finalized_capability", None)
            is not _FRV_SOURCE_FINALIZED_CAPABILITY
        ):
            raise _build_hag_error("SOURCE_RESULT_INVALID")
        try:
            validate_finalized_frv_source_provider_result(self._source_result)
        except Exception as exc:  # noqa: BLE001 - validator failure boundary
            raise _build_hag_error("SOURCE_RESULT_INVALID") from exc
        if type(self._reference) is not FrvSourceProviderReference:
            raise _build_hag_error("SOURCE_REFERENCE_INVALID")
        try:
            validate_configuration(self._frv_configuration)
        except Exception as exc:  # noqa: BLE001 - validator failure boundary
            raise _build_hag_error("PROVIDER_CONFIG_INVALID") from exc
        try:
            validate_frv_hag_provider_config(self._provider_config)
        except Exception as exc:  # noqa: BLE001 - config validation boundary
            raise _build_hag_error("PROVIDER_CONFIG_INVALID") from exc
        try:
            self._reconstructed_config = _reconstruct_frv_source_provider_config(
                self._source_result.proof
            )
        except FrvHagLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconstruction boundary
            raise _build_hag_error("PROVIDER_CONFIG_INVALID") from exc
        try:
            validate_frv_source_provider_reference(
                self._reference,
                provider_config=self._reconstructed_config,
                expected_configuration_digest=self._frv_configuration.configuration_digest,
            )
        except Exception as exc:  # noqa: BLE001 - reference binding boundary
            raise _build_hag_error("SOURCE_REFERENCE_INVALID") from exc
        source_proof = self._source_result.proof
        if source_proof.provider_role is not FrvA2ProviderRole.SOURCE_SCAN:
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
        if source_proof.provider_key != FRV_SOURCE_PROVIDER_KEY:
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
        if source_proof.domain_kind is not FrvA2DomainKind.FULL_SOURCE:
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
        if source_proof.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
            raise _build_hag_error("COORDINATE_FRAME_INVALID")
        if source_proof.units != "m":
            raise _build_hag_error("UNITS_INVALID")
        if source_proof.transform_declaration != "identity":
            raise _build_hag_error("TRANSFORM_INVALID")
        if source_proof.source_point_count <= 0:
            raise _build_hag_error("POINT_COUNT_MISMATCH")
        if FrvSourcePassThroughDimension.CLASSIFICATION not in (
            source_proof.pass_through_dimensions
        ):
            raise _build_hag_error("CLASSIFICATION_MISSING")
        if "classification" not in source_proof.header.dimensions:
            raise _build_hag_error("CLASSIFICATION_MISSING")
        source_identity = self._reference.source_identity
        result_identity = self._source_result.source_identity
        if (
            source_identity.source_content_identity_digest
            != result_identity.source_content_identity_digest
            or source_identity.source_identity_record_digest
            != result_identity.source_identity_record_digest
        ):
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[FrvHagChunk]:
        if self._state is not _CREATED:
            raise _build_hag_error("REPEATED_ITERATOR")
        self._state = _ITERATING
        try:
            self._run_pass1()
        except FrvHagLoadError:
            self._abort()
            raise
        except Exception as exc:
            self._abort()
            raise _build_hag_error("SOURCE_PASS_INCOMPLETE") from exc
        return self._stream_hag_chunks()

    def _stream_hag_chunks(self):
        try:
            self._hag_pcd1 = hashlib.sha256()
            self._hag_pcd1.update(b"FRVHAGPCD1\0")
            self._hag_pcd1.update(
                self._source_result.proof.source_content_identity_digest.encode("utf-8")
            )
            self._hag_pcd1.update(b"\0")
            self._hag_pcd1.update(
                self._source_result.proof.source_identity_record_digest.encode("utf-8")
            )
            self._hag_pcd1.update(b"\0")
            self._hag_pcd1.update(self._provider_config.provider_config_digest.encode("utf-8"))
            self._hag_pcd1.update(b"\0")
            self._hag_pcd1.update(struct.pack("<Q", self._source_point_count))
            session2 = load_frv_source(
                self._reference,
                frv_configuration=self._frv_configuration,
                provider_config=self._reconstructed_config,
                chunk_size=self._source_chunk_size,
            )
            self._active_source_sessions.append(session2)
            try:
                with session2:
                    for source_chunk in session2.iter_chunks():
                        if source_chunk.source_id != self._expected_source_id:
                            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
                        if (
                            self._pass1_source_id is not None
                            and source_chunk.source_id != self._pass1_source_id
                        ):
                            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
                        hag_chunk = self._build_hag_chunk(source_chunk)
                        self._feed_hag_pcd1(hag_chunk)
                        yield hag_chunk
                    result2 = session2.finalize()
                self._check_source_result_matches(result2)
                self._result2 = result2
            finally:
                if session2 in self._active_source_sessions:
                    self._active_source_sessions.remove(session2)
            if self._expected_hag_index != self._source_point_count:
                raise _build_hag_error("INDEX_LOST")
            self._state = _STREAM_COMPLETE
        except GeneratorExit:
            self._abort()
            raise
        except FrvHagLoadError:
            self._abort()
            raise
        except Exception as exc:
            self._abort()
            raise _build_hag_error("SOURCE_PASS_INCOMPLETE") from exc

    def finalize(self) -> FrvHagProviderResult:
        if self._state in (_CREATED, _ITERATING, _ABORTED):
            raise _build_hag_error("PREMATURE_FINALIZE")
        if self._state is not _STREAM_COMPLETE:
            raise _build_hag_error("REPEATED_FINALIZE")
        try:
            proof = self._build_proof()
            result = _create_finalized_frv_hag_provider_result(
                proof,
                self._source_result.source_identity,
                capability=_FRV_HAG_FINALIZED_CAPABILITY,
            )
            self._result = result
            self._state = _FINALIZED
            return result
        except FrvHagLoadError:
            self._abort()
            raise
        except ValueError as exc:
            self._abort()
            raise _build_hag_error("SOURCE_BINDING_MISMATCH") from exc
        except Exception as exc:
            self._abort()
            raise _build_hag_error("DIGEST_MISMATCH") from exc

    def close(self) -> None:
        if self._state is _CLOSED:
            return
        was_finalized = self._state is _FINALIZED
        had_prior_failure = self._had_failure
        if not was_finalized:
            self._state = _ABORTED
        cleanup_error = self._close_source_sessions()
        self._state = _CLOSED
        if cleanup_error is not None and not had_prior_failure:
            if was_finalized:
                raise _build_hag_error("CLEANUP_FAILURE") from cleanup_error
            raise _build_hag_error("CLEANUP_FAILURE") from cleanup_error

    def __enter__(self) -> FrvHagLoadSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:  # noqa: BLE001 - must not mask the active exception
                pass
        return False

    # ------------------------------------------------------------------
    # Pass 1: sparse ground grid
    # ------------------------------------------------------------------

    def _run_pass1(self) -> None:
        min_x = float(self._source_result.proof.source_bounds_min_xyz[0])
        min_y = float(self._source_result.proof.source_bounds_min_xyz[1])
        grid: dict[tuple[int, int], tuple[float, int]] = {}
        session1 = load_frv_source(
            self._reference,
            frv_configuration=self._frv_configuration,
            provider_config=self._reconstructed_config,
            chunk_size=self._source_chunk_size,
        )
        self._active_source_sessions.append(session1)
        try:
            with session1:
                for chunk in session1.iter_chunks():
                    if chunk.source_id != self._expected_source_id:
                        raise _build_hag_error("SOURCE_BINDING_MISMATCH")
                    if self._pass1_source_id is None:
                        self._pass1_source_id = chunk.source_id
                    elif chunk.source_id != self._pass1_source_id:
                        raise _build_hag_error("SOURCE_BINDING_MISMATCH")
                    self._accumulate_grid(chunk, grid, min_x, min_y)
                result1 = session1.finalize()
            self._check_source_result_matches(result1)
        finally:
            if session1 in self._active_source_sessions:
                self._active_source_sessions.remove(session1)
        self._grid = grid
        if not grid:
            raise _build_hag_error("NO_GROUND")

    def _accumulate_grid(
        self,
        source_chunk: Any,
        grid: dict[tuple[int, int], tuple[float, int]],
        min_x: float,
        min_y: float,
    ) -> None:
        if "classification" not in source_chunk.attributes:
            raise _build_hag_error("CLASSIFICATION_MISSING")
        classification = source_chunk.attributes["classification"]
        if classification.dtype != np.dtype("|u1"):
            raise _build_hag_error("CLASSIFICATION_DTYPE")
        cell_size = self._provider_config.cell_size_m
        for row in range(source_chunk.point_count):
            if int(classification[row]) != 2:
                continue
            x = float(source_chunk.xyz[row, 0])
            y = float(source_chunk.xyz[row, 1])
            z = float(source_chunk.xyz[row, 2])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                raise _build_hag_error("NONFINITE_GROUND")
            cell_ix = math.floor((x - min_x) / cell_size)
            cell_iy = math.floor((y - min_y) / cell_size)
            ground_z = _normalize_zero(z)
            key = (cell_ix, cell_iy)
            if key in grid:
                min_z, count = grid[key]
                if ground_z < min_z:
                    min_z = ground_z
                grid[key] = (min_z, count + 1)
            else:
                grid[key] = (ground_z, 1)
        self._ground_point_count += int(np.count_nonzero(classification == 2))

    # ------------------------------------------------------------------
    # Pass 2: HAG evaluation
    # ------------------------------------------------------------------

    def _build_hag_chunk(self, source_chunk: Any) -> FrvHagChunk:
        point_count = source_chunk.point_count
        original = np.ascontiguousarray(
            source_chunk.original_index.astype("<i8", copy=True)
        )
        original.setflags(write=False)
        ground_support = np.zeros(point_count, dtype=bool)
        support_cell_count = np.zeros(point_count, dtype="<u2")
        ground_z_m = np.zeros(point_count, dtype="<f8")
        raw_hag_m = np.zeros(point_count, dtype="<f8")
        clamped_hag_m = np.zeros(point_count, dtype="<f8")
        hag_state = np.zeros(point_count, dtype="|u1")
        reason_flags = np.zeros(point_count, dtype="<u4")
        quality_flags = np.zeros(point_count, dtype="<u4")
        for row in range(point_count):
            evaluated = self._evaluate_row(source_chunk, row)
            (
                supported,
                count,
                ground_z,
                raw_hag,
                clamped,
                state,
                reason,
                quality,
            ) = evaluated
            ground_support[row] = supported
            support_cell_count[row] = count
            ground_z_m[row] = ground_z
            raw_hag_m[row] = raw_hag
            clamped_hag_m[row] = clamped
            hag_state[row] = state
            reason_flags[row] = reason
            quality_flags[row] = quality
        for array in (
            ground_support,
            support_cell_count,
            ground_z_m,
            raw_hag_m,
            clamped_hag_m,
            hag_state,
            reason_flags,
            quality_flags,
        ):
            array.setflags(write=False)
        return FrvHagChunk(
            source_id=source_chunk.source_id,
            chunk_id=source_chunk.chunk_id,
            start_index=source_chunk.start_index,
            point_count=point_count,
            original_index=original,
            ground_support=ground_support,
            support_cell_count=support_cell_count,
            ground_z_m=ground_z_m,
            raw_hag_m=raw_hag_m,
            clamped_hag_m=clamped_hag_m,
            hag_evidence_state=hag_state,
            reason_flags=reason_flags,
            quality_flags=quality_flags,
            is_last=source_chunk.is_last,
        )

    def _evaluate_row(self, source_chunk: Any, row: int) -> tuple:
        x = float(source_chunk.xyz[row, 0])
        y = float(source_chunk.xyz[row, 1])
        z = float(source_chunk.xyz[row, 2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise _build_hag_error("NONFINITE_XYZ")
        candidates = self._find_support_cells(x, y)
        if len(candidates) < self._provider_config.min_support_cells:
            return (
                False,
                0,
                0.0,
                0.0,
                0.0,
                int(FrvHagEvidenceState.HAG_MISSING),
                int(FrvReasonFlag.INSUFFICIENT_EVIDENCE | FrvReasonFlag.HAG_MISSING),
                0,
            )
        candidates = candidates[: self._provider_config.max_support_cells]
        if candidates[0][0] == 0.0:
            ground_z = candidates[0][3]
        else:
            numerator = 0.0
            denominator = 0.0
            for squared, _ix, _iy, cell_min_z in candidates:
                weight = 1.0 / squared
                if not math.isfinite(weight):
                    raise _build_hag_error("NONFINITE_INTERPOLATION")
                numerator += weight * cell_min_z
                denominator += weight
                if not math.isfinite(numerator) or not math.isfinite(denominator):
                    raise _build_hag_error("NONFINITE_INTERPOLATION")
            if denominator == 0.0:
                raise _build_hag_error("NONFINITE_INTERPOLATION")
            ground_z = numerator / denominator
            if not math.isfinite(ground_z):
                raise _build_hag_error("NONFINITE_INTERPOLATION")
        raw_hag = z - ground_z
        clamped = max(raw_hag, 0.0)
        tolerance = self._provider_config.negative_hag_tolerance_m
        if raw_hag > 0.0:
            state = FrvHagEvidenceState.HAG_VALID_POSITIVE
            reason = 0
            quality = 0
        elif raw_hag == 0.0:
            state = FrvHagEvidenceState.HAG_VALID_NEAR_ZERO
            reason = 0
            quality = 0
        elif raw_hag >= -tolerance:
            state = FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE
            reason = 0
            quality = 0
        else:
            state = FrvHagEvidenceState.HAG_INVALID
            reason = int(FrvReasonFlag.HAG_INVALID)
            quality = int(FrvQualityFlag.PROVIDER_INVALID)
        return (
            True,
            len(candidates),
            _normalize_zero(ground_z),
            _normalize_zero(raw_hag),
            _normalize_zero(clamped),
            int(state),
            reason,
            quality,
        )

    def _find_support_cells(self, x: float, y: float) -> list[tuple[float, int, int, float]]:
        min_x = float(self._source_result.proof.source_bounds_min_xyz[0])
        min_y = float(self._source_result.proof.source_bounds_min_xyz[1])
        cell_size = self._provider_config.cell_size_m
        radius = self._provider_config.support_radius_m
        ix_lo = math.ceil((x - radius - min_x) / cell_size - 0.5)
        ix_hi = math.floor((x + radius - min_x) / cell_size - 0.5)
        iy_lo = math.ceil((y - radius - min_y) / cell_size - 0.5)
        iy_hi = math.floor((y + radius - min_y) / cell_size - 0.5)
        radius_sq = radius * radius
        candidates: list[tuple[float, int, int, float]] = []
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                key = (ix, iy)
                if key not in self._grid:
                    continue
                cell_min_z, _count = self._grid[key]
                center_x = min_x + (ix + 0.5) * cell_size
                center_y = min_y + (iy + 0.5) * cell_size
                dx = x - center_x
                dy = y - center_y
                squared = dx * dx + dy * dy
                if not math.isfinite(squared):
                    raise _build_hag_error("NONFINITE_INTERPOLATION")
                if squared <= radius_sq:
                    candidates.append((squared, ix, iy, cell_min_z))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return candidates

    # ------------------------------------------------------------------
    # Digest feed and counts
    # ------------------------------------------------------------------

    def _feed_hag_pcd1(self, hag_chunk: FrvHagChunk) -> None:
        assert self._hag_pcd1 is not None
        for row in range(hag_chunk.point_count):
            index = int(hag_chunk.original_index[row])
            if index < 0:
                raise _build_hag_error("INDEX_OUT_OF_RANGE")
            if index != self._expected_hag_index:
                if index < self._expected_hag_index:
                    raise _build_hag_error("INDEX_DUPLICATE")
                raise _build_hag_error("INDEX_LOST")
            self._expected_hag_index += 1
            supported = bool(hag_chunk.ground_support[row])
            count = int(hag_chunk.support_cell_count[row])
            state = int(hag_chunk.hag_evidence_state[row])
            self._hag_pcd1.update(struct.pack("<Q", index))
            self._hag_pcd1.update(struct.pack("<B", int(supported)))
            self._hag_pcd1.update(struct.pack("<H", count))
            self._hag_pcd1.update(struct.pack("<d", float(hag_chunk.ground_z_m[row])))
            self._hag_pcd1.update(struct.pack("<d", float(hag_chunk.raw_hag_m[row])))
            self._hag_pcd1.update(struct.pack("<d", float(hag_chunk.clamped_hag_m[row])))
            self._hag_pcd1.update(struct.pack("<B", state))
            self._hag_pcd1.update(struct.pack("<I", int(hag_chunk.reason_flags[row])))
            self._hag_pcd1.update(struct.pack("<I", int(hag_chunk.quality_flags[row])))
            if supported:
                self._supported_count += 1
            else:
                self._unsupported_count += 1
            if state == int(FrvHagEvidenceState.HAG_VALID_POSITIVE):
                self._valid_positive_count += 1
            elif state == int(FrvHagEvidenceState.HAG_VALID_NEAR_ZERO):
                self._valid_near_zero_count += 1
            elif state == int(FrvHagEvidenceState.HAG_WITHIN_NEGATIVE_TOLERANCE):
                self._within_negative_count += 1
            elif state == int(FrvHagEvidenceState.HAG_MISSING):
                self._missing_count += 1
            elif state == int(FrvHagEvidenceState.HAG_INVALID):
                self._invalid_count += 1

    # ------------------------------------------------------------------
    # Source binding comparison and proof
    # ------------------------------------------------------------------

    def _check_source_result_matches(self, result: FrvSourceProviderResult) -> None:
        expected = self._source_result
        if result.proof.to_dict() != expected.proof.to_dict():
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
        if result.source_identity != expected.source_identity:
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")
        if (
            to_frv_a2_provider_binding_evidence(result).binding_digest
            != to_frv_a2_provider_binding_evidence(expected).binding_digest
        ):
            raise _build_hag_error("SOURCE_BINDING_MISMATCH")

    def _build_proof(self) -> FrvHagProviderProof:
        assert self._result2 is not None
        source_proof = self._source_result.proof
        source_proof_digest = frv_digest(
            {
                "digest_domain": FRV_HAG_SOURCE_PROOF_DIGEST_DOMAIN,
                "source_proof": source_proof.to_dict(),
            }
        )
        source_binding_digest = to_frv_a2_provider_binding_evidence(
            self._source_result
        ).binding_digest
        components = source_proof.artifact_components
        artifact_reference_id = _derive_artifact_reference_id(
            FrvA2ProviderRole.HAG, FRV_HAG_PROVIDER_KEY, components
        )
        grid_cells = tuple(
            (ix, iy, min_z, count)
            for (ix, iy), (min_z, count) in sorted(self._grid.items())
        )
        ground_cell_inventory_digest = frv_hag_ground_cell_inventory_digest(
            source_content_identity_digest=source_proof.source_content_identity_digest,
            source_identity_record_digest=source_proof.source_identity_record_digest,
            provider_config_digest=self._provider_config.provider_config_digest,
            ground_point_count=self._ground_point_count,
            cells=grid_cells,
        )
        assert self._hag_pcd1 is not None
        proof = FrvHagProviderProof.__new__(FrvHagProviderProof)
        object.__setattr__(proof, "schema_version", FRV_HAG_PROOF_SCHEMA_VERSION)
        object.__setattr__(proof, "provider_key", FRV_HAG_PROVIDER_KEY)
        object.__setattr__(proof, "provider_role", FrvA2ProviderRole.HAG)
        object.__setattr__(proof, "interface_version", FRV_HAG_PROVIDER_INTERFACE_VERSION)
        object.__setattr__(proof, "algorithm_version", FRV_HAG_ALGORITHM_VERSION)
        object.__setattr__(
            proof, "configuration_digest", self._frv_configuration.configuration_digest
        )
        object.__setattr__(
            proof, "provider_config_digest", self._provider_config.provider_config_digest
        )
        object.__setattr__(
            proof, "source_artifact_reference_id", source_proof.artifact_reference_id
        )
        object.__setattr__(proof, "artifact_reference_id", artifact_reference_id)
        object.__setattr__(proof, "artifact_components", components)
        object.__setattr__(
            proof,
            "source_content_identity_digest",
            source_proof.source_content_identity_digest,
        )
        object.__setattr__(
            proof,
            "source_identity_record_digest",
            source_proof.source_identity_record_digest,
        )
        object.__setattr__(proof, "source_proof_digest", source_proof_digest)
        object.__setattr__(proof, "source_binding_digest", source_binding_digest)
        object.__setattr__(proof, "source_point_count", self._source_point_count)
        object.__setattr__(proof, "domain_kind", FrvA2DomainKind.FULL_SOURCE)
        object.__setattr__(proof, "domain_count", self._source_point_count)
        object.__setattr__(
            proof,
            "original_index_domain_digest",
            source_proof.original_index_domain_digest,
        )
        object.__setattr__(
            proof, "hag_point_content_digest", self._hag_pcd1.hexdigest()
        )
        object.__setattr__(
            proof, "ground_cell_inventory_digest", ground_cell_inventory_digest
        )
        object.__setattr__(proof, "ground_point_count", self._ground_point_count)
        object.__setattr__(proof, "supported_point_count", self._supported_count)
        object.__setattr__(proof, "unsupported_point_count", self._unsupported_count)
        object.__setattr__(
            proof,
            "negative_tolerance_exceeded_point_count",
            self._invalid_count,
        )
        object.__setattr__(
            proof, "hag_valid_positive_count", self._valid_positive_count
        )
        object.__setattr__(
            proof, "hag_valid_near_zero_count", self._valid_near_zero_count
        )
        object.__setattr__(
            proof, "hag_within_negative_tolerance_count", self._within_negative_count
        )
        object.__setattr__(proof, "hag_missing_count", self._missing_count)
        object.__setattr__(proof, "hag_invalid_count", self._invalid_count)
        object.__setattr__(proof, "hag_out_of_domain_count", 0)
        object.__setattr__(proof, "coordinate_frame", "LOCAL_CARTESIAN_XYZ")
        object.__setattr__(proof, "units", "m")
        object.__setattr__(proof, "origin_xyz", source_proof.origin_xyz)
        object.__setattr__(proof, "transform_declaration", "identity")
        object.__setattr__(
            proof, "pass_through_dimensions", source_proof.pass_through_dimensions
        )
        object.__setattr__(proof, "quality_flags", 0)
        object.__setattr__(proof, "reason_flags", 0)
        object.__setattr__(proof, "error_codes", ())
        object.__setattr__(proof, "proof_digest", proof.computed_digest())
        return proof

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _mark_aborted(self) -> None:
        if self._state not in (_FINALIZED, _CLOSED, _ABORTED):
            self._state = _ABORTED

    def _abort(self) -> None:
        self._mark_aborted()
        self._close_source_sessions()
        self._had_failure = True

    def _close_source_sessions(self) -> BaseException | None:
        first_error = None
        sessions = list(self._active_source_sessions)
        self._active_source_sessions.clear()
        for session in sessions:
            try:
                session.close()
            except Exception as exc:  # noqa: BLE001 - cleanup path
                if first_error is None:
                    first_error = exc
        return first_error


def load_frv_hag(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvHagProviderConfig,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvHagLoadSession:
    """Open a single-use two-pass P-HAG session (private implementation)."""

    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(frv_configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")
    if type(provider_config) is not FrvHagProviderConfig:
        raise TypeError("provider_config must be a FrvHagProviderConfig")
    if type(source_chunk_size) is not int:
        raise TypeError("source_chunk_size must be a built-in int")
    if source_chunk_size <= 0:
        raise ValueError("source_chunk_size must be positive")
    return _FrvHagLoadSession(
        reference, source_result, frv_configuration, provider_config, source_chunk_size
    )
