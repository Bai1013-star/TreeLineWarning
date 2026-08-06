"""FRV-A2 P-GEOMETRY runtime provider (PLAN-FROZEN).

P-GEOMETRY consumes a finalized P-SOURCE result and runs exactly two
independent ``load_frv_source`` sessions.  Pass G1 builds an on-disk spatial
cell spool (``<qddd`` records anchored at the source proof minimum bounds).
Pass G2 computes the multiscale neighborhood/PCA/vertical features and the
mutual structural-support edges, emitting one geometry chunk per source chunk
over the complete FULL_SOURCE domain.  No direct LAS access, no third pass, no
full source materialization.
"""

from __future__ import annotations

import heapq
import math
import os
import shutil
import struct
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import numpy as np
from core.contracts.frv import (
    FrvConfiguration,
    FrvErrorCode,
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
from core.contracts.frv_geometry import (
    _FRV_GEOMETRY_FINALIZED_CAPABILITY,
    _FRV_GEOMETRY_SOURCE_PROOF_DIGEST_DOMAIN,
    FRV_GEOMETRY_ALGORITHM_VERSION,
    FRV_GEOMETRY_CACHE_BYTES,
    FRV_GEOMETRY_PROOF_SCHEMA_VERSION,
    FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION,
    FRV_GEOMETRY_PROVIDER_KEY,
    FrvGeometryChunk,
    FrvGeometryEvidenceFlag,
    FrvGeometryEvidenceState,
    FrvGeometryLoadError,
    FrvGeometryLoadSession,
    FrvGeometryProviderConfig,
    FrvGeometryProviderProof,
    FrvGeometryProviderResult,
    FrvGeometryQualityFlag,
    FrvGeometryReasonFlag,
    _create_finalized_frv_geometry_provider_result,
    frv_geometry_point_content_digest,
    validate_frv_geometry_provider_config,
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
_INDEXING = "INDEXING"
_INDEX_READY = "INDEX_READY"
_ITERATING = "ITERATING"
_STREAM_COMPLETE = "STREAM_COMPLETE"
_FINALIZED = "FINALIZED"
_ABORTED = "ABORTED"
_CLOSED = "CLOSED"

_SCALE_COUNT = 3
_SCALE_MEDIUM = 1
_SCALE_LARGE = 2
_RECORD_BYTES = 32

# Frozen plan section 3.8 mapping keys -> failure-record tuple.  Only existing
# A2 enums/flags are used.
_GEOMETRY_FAILURE_MAP: dict[
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
    "PROVIDER_CONFIG_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "provider_config",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "SOURCE_IDENTITY_MISMATCH": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "source_identity",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "NONFINITE_XYZ": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "xyz",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "INDEX_CELL_LIMIT": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "index_cell",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "SCRATCH_STORE": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "scratch_store",
        FrvErrorCode.FRV_PROVIDER_INVALID,
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
    "SOURCE_PASS_INCOMPLETE": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "source_pass",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "POINT_COUNT_MISMATCH": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "source_pass",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "PCA_INVALID": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "pca",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "EIGENVALUE_INVALID": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "eigenvalue",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
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


def _build_geometry_error(key: str) -> FrvGeometryLoadError:
    phase, field_name, error_code, reason_flags, quality_flag = _GEOMETRY_FAILURE_MAP[key]
    detail = (
        f"{phase.value}:{FrvA2ProviderRole.GEOMETRY.value}:{FRV_GEOMETRY_PROVIDER_KEY}:"
        f"{field_name}:{error_code.value}:{int(reason_flags)}"
    )
    record = FrvA2FailureRecord(
        phase,
        FrvA2ProviderRole.GEOMETRY,
        FRV_GEOMETRY_PROVIDER_KEY,
        field_name,
        error_code,
        int(reason_flags),
        detail,
    )
    return FrvGeometryLoadError(failure_record=record, quality_flags=int(quality_flag))


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
        raise _build_geometry_error("PROVIDER_CONFIG_INVALID")
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=pass_through_dimensions,
        provider_config_digest=provider_config_digest,
    )


def _heap_push(
    heap: list[tuple[Any, ...]],
    key: tuple[float, int],
    extra: tuple,
    capacity: int,
) -> None:
    """Keep the ``capacity`` smallest ``(d2, index)`` keys in a max-heap.

    The heap stores negated keys so that the standard min-heap acts as a
    max-heap over the total order ``(d2, index)``.  The surviving items are
    exactly the ``capacity`` smallest pairs, byte-identical to a full sort.
    """

    item = (-key[0], -key[1]) + tuple(extra)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif key < (-heap[0][0], -heap[0][1]):
        heapq.heapreplace(heap, item)


def _heap_items(heap: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Recover the ascending ``(d2, index, *extra)`` items from a max-heap."""
    return sorted((-item[0], -item[1]) + tuple(item[2:]) for item in heap)


def _sign_canonicalized(vector: Any) -> tuple[float, float, float]:
    """Flip each eigenvector so its largest-magnitude component is positive.

    Absolute-value ties resolve to the first (x, y, z) component, which is the
    deterministic ``numpy.argmax`` tie-break.
    """

    components = [float(vector[0]), float(vector[1]), float(vector[2])]
    magnitudes = [abs(component) for component in components]
    axis = 0
    for index in (1, 2):
        if magnitudes[index] > magnitudes[axis]:
            axis = index
    if components[axis] < 0.0:
        components = [-component for component in components]
    return (components[0], components[1], components[2])


class _CellSpool:
    """On-disk spatial cell spool with an LRU handle cache and LRU byte cache.

    Records are fixed 32-byte ``<qddd`` little-endian tuples
    (``original_index``, ``x``, ``y``, ``z``).  A record is appended to exactly
    one cell file keyed by ``(ix, iy, iz)``; records within a cell are
    naturally increasing by ``original_index``.  Cache hit/miss never changes
    results; the byte cache budget is the frozen 512 MiB limit.
    """

    def __init__(
        self,
        directory: Path,
        min_xyz: tuple[float, float, float],
        cell_size: float,
    ) -> None:
        self._directory = directory
        self._min_x, self._min_y, self._min_z = min_xyz
        self._cell_size = cell_size
        self._counts: dict[tuple[int, int, int], int] = {}
        self._write_handles: dict[tuple[int, int, int], Any] = {}
        self._write_order: list[tuple[int, int, int]] = []
        self._max_open_handles = 32
        self._read_cache: dict[tuple[int, int, int], bytes] = {}
        self._read_order: list[tuple[int, int, int]] = []
        self._cache_bytes = 0
        self._cache_budget = FRV_GEOMETRY_CACHE_BYTES

    def cell_key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        ix = math.floor((x - self._min_x) / self._cell_size)
        iy = math.floor((y - self._min_y) / self._cell_size)
        iz = math.floor((z - self._min_z) / self._cell_size)
        return (ix, iy, iz)

    def _cell_path(self, key: tuple[int, int, int]) -> Path:
        return self._directory / f"cell_{key[0]}_{key[1]}_{key[2]}.bin"

    def append(self, index: int, x: float, y: float, z: float) -> None:
        key = self.cell_key(x, y, z)
        self._counts[key] = self._counts.get(key, 0) + 1
        handle = self._write_handle(key)
        handle.write(struct.pack("<qddd", index, x, y, z))

    def cell_count(self, key: tuple[int, int, int]) -> int:
        return self._counts.get(key, 0)

    def _write_handle(self, key: tuple[int, int, int]):
        if key not in self._write_handles:
            if len(self._write_handles) >= self._max_open_handles:
                self._evict_write_handle()
            handle = open(self._cell_path(key), "ab")
            self._write_handles[key] = handle
        if key in self._write_order:
            self._write_order.remove(key)
        self._write_order.append(key)
        return self._write_handles[key]

    def _evict_write_handle(self) -> None:
        oldest = self._write_order[0]
        self._write_order.remove(oldest)
        handle = self._write_handles.pop(oldest)
        try:
            handle.close()
        except OSError:  # pragma: no cover - cleanup path
            pass

    def close_writes(self) -> None:
        for key in list(self._write_handles):
            handle = self._write_handles.pop(key)
            try:
                handle.close()
            except OSError:  # pragma: no cover - cleanup path
                pass
        self._write_order.clear()

    def inventory(self) -> tuple[tuple[int, int, int], ...]:
        return tuple(sorted(self._counts))

    def read_cell(self, key: tuple[int, int, int]):
        data = self._read_cell_bytes(key)
        for offset in range(0, len(data), _RECORD_BYTES):
            index, x, y, z = struct.unpack("<qddd", data[offset : offset + _RECORD_BYTES])
            yield index, x, y, z

    def _read_cell_bytes(self, key: tuple[int, int, int]) -> bytes:
        if key in self._read_cache:
            self._read_order.remove(key)
            self._read_order.append(key)
            return self._read_cache[key]
        path = self._cell_path(key)
        if not path.exists():
            return b""
        data = path.read_bytes()
        if len(data) <= self._cache_budget:
            self._read_cache[key] = data
            self._cache_bytes += len(data)
            self._read_order.append(key)
            while self._cache_bytes > self._cache_budget and len(self._read_order) > 1:
                stale = self._read_order.pop(0)
                self._cache_bytes -= len(self._read_cache.pop(stale))
        return data


class _FrvGeometryLoadSession:
    """Single-use two-pass P-GEOMETRY session (private; not exported)."""

    def __init__(
        self,
        reference: FrvSourceProviderReference,
        source_result: FrvSourceProviderResult,
        frv_configuration: FrvConfiguration,
        provider_config: FrvGeometryProviderConfig,
        scratch_dir: Path,
        source_chunk_size: int,
    ) -> None:
        self._reference = reference
        self._source_result = source_result
        self._frv_configuration = frv_configuration
        self._provider_config = provider_config
        self._scratch_dir = scratch_dir
        self._source_chunk_size = source_chunk_size
        self._state = _CREATED
        self._had_failure = False
        self._result: FrvGeometryProviderResult | None = None
        self._result2: FrvSourceProviderResult | None = None
        self._source_point_count = int(source_result.proof.source_point_count)
        self._expected_source_id = str(source_result.source_identity.source_id)
        self._pass1_source_id: str | None = None
        self._spool_dir: Path | None = None
        self._spool: _CellSpool | None = None
        self._pending_chunks: list[tuple] = []
        self._support_candidates: list[list[tuple[float, int]]] = []
        self._support_sets: list[frozenset[int]] = []
        self._support_valid: list[bool] = []
        self._support_canopy: list[bool] = []
        self._support_direction: list[tuple[float, float, float]] = []
        self._active_source_sessions: list[Any] = []
        self._geometry_point_content_digest: str | None = None
        self._valid_count = 0
        self._insufficient_count = 0
        self._degenerate_count = 0
        self._neighborhood_sufficient_count = 0
        self._density_supported_count = 0
        self._canopy_branching_compatible_count = 0
        self._woody_column_compatible_count = 0
        self._ground_surface_compatible_count = 0
        self._vertical_continuity_count = 0
        self._neighbor_cap_reached_count = 0
        self._support_neighbor_cap_reached_count = 0
        self._scratch_checked = False
        self._pre_pass_validate()

    # ------------------------------------------------------------------
    # Pre-pass validation (frozen first-failure order)
    # ------------------------------------------------------------------

    def _pre_pass_validate(self) -> None:
        if type(self._source_result) is not FrvSourceProviderResult:
            raise _build_geometry_error("SOURCE_RESULT_INVALID")
        if (
            getattr(self._source_result, "_finalized_capability", None)
            is not _FRV_SOURCE_FINALIZED_CAPABILITY
        ):
            raise _build_geometry_error("SOURCE_RESULT_INVALID")
        try:
            validate_finalized_frv_source_provider_result(self._source_result)
        except Exception as exc:  # noqa: BLE001 - validator failure boundary
            raise _build_geometry_error("SOURCE_RESULT_INVALID") from exc
        if type(self._reference) is not FrvSourceProviderReference:
            raise _build_geometry_error("SOURCE_REFERENCE_INVALID")
        try:
            validate_configuration(self._frv_configuration)
        except Exception as exc:  # noqa: BLE001 - validator failure boundary
            raise _build_geometry_error("PROVIDER_CONFIG_INVALID") from exc
        try:
            validate_frv_geometry_provider_config(self._provider_config)
        except Exception as exc:  # noqa: BLE001 - config validation boundary
            raise _build_geometry_error("PROVIDER_CONFIG_INVALID") from exc
        if (
            self._provider_config.interface_version
            != self._frv_configuration.geometry_provider_interface_version
            or self._provider_config.provider_config_digest
            != self._frv_configuration.geometry_provider_config_digest
        ):
            raise _build_geometry_error("PROVIDER_CONFIG_INVALID")
        try:
            self._reconstructed_config = _reconstruct_frv_source_provider_config(
                self._source_result.proof
            )
        except FrvGeometryLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconstruction boundary
            raise _build_geometry_error("PROVIDER_CONFIG_INVALID") from exc
        try:
            validate_frv_source_provider_reference(
                self._reference,
                provider_config=self._reconstructed_config,
                expected_configuration_digest=self._frv_configuration.configuration_digest,
            )
        except Exception as exc:  # noqa: BLE001 - reference binding boundary
            raise _build_geometry_error("SOURCE_REFERENCE_INVALID") from exc
        source_proof = self._source_result.proof
        if source_proof.provider_role is not FrvA2ProviderRole.SOURCE_SCAN:
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
        if source_proof.provider_key != FRV_SOURCE_PROVIDER_KEY:
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
        if source_proof.domain_kind is not FrvA2DomainKind.FULL_SOURCE:
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
        if source_proof.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
        if source_proof.units != "m":
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
        if source_proof.transform_declaration != "identity":
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
        if source_proof.source_point_count <= 0:
            raise _build_geometry_error("POINT_COUNT_MISMATCH")
        source_identity = self._reference.source_identity
        result_identity = self._source_result.source_identity
        if (
            source_identity.source_content_identity_digest
            != result_identity.source_content_identity_digest
            or source_identity.source_identity_record_digest
            != result_identity.source_identity_record_digest
        ):
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[FrvGeometryChunk]:
        if self._state is not _CREATED:
            raise _build_geometry_error("REPEATED_ITERATOR")
        self._state = _INDEXING
        try:
            self._run_pass1()
        except FrvGeometryLoadError:
            self._abort()
            raise
        except Exception as exc:
            self._abort()
            raise _build_geometry_error("SOURCE_PASS_INCOMPLETE") from exc
        if self._state is not _INDEX_READY:
            self._abort()
            raise _build_geometry_error("SOURCE_PASS_INCOMPLETE")
        self._state = _ITERATING
        return self._stream_geometry_chunks()

    def _stream_geometry_chunks(self) -> Iterator[FrvGeometryChunk]:
        try:
            self._run_pass2_source()
            self._resolve_support_edges()
            assembled = [self._assemble_geometry_chunk(data) for data in self._pending_chunks]
            self._check_geometry_stream_continuity(assembled)
            self._geometry_point_content_digest = frv_geometry_point_content_digest(
                source_content_identity_digest=self._source_result.proof.source_content_identity_digest,
                source_identity_record_digest=self._source_result.proof.source_identity_record_digest,
                provider_config_digest=self._provider_config.provider_config_digest,
                source_point_count=self._source_point_count,
                chunks=assembled,
            )
            for chunk in assembled:
                self._accumulate_counts(chunk)
                yield chunk
            self._state = _STREAM_COMPLETE
        except GeneratorExit:
            self._abort()
            raise
        except FrvGeometryLoadError:
            self._abort()
            raise
        except Exception as exc:
            self._abort()
            raise _build_geometry_error("SOURCE_PASS_INCOMPLETE") from exc

    def finalize(self) -> FrvGeometryProviderResult:
        if self._state in (_CREATED, _INDEXING, _INDEX_READY, _ITERATING, _ABORTED):
            raise _build_geometry_error("PREMATURE_FINALIZE")
        if self._state is not _STREAM_COMPLETE:
            raise _build_geometry_error("REPEATED_FINALIZE")
        try:
            proof = self._build_proof()
            result = _create_finalized_frv_geometry_provider_result(
                proof,
                self._source_result.source_identity,
                capability=_FRV_GEOMETRY_FINALIZED_CAPABILITY,
            )
            self._result = result
            self._state = _FINALIZED
            return result
        except FrvGeometryLoadError:
            self._abort()
            raise
        except ValueError as exc:
            self._abort()
            raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH") from exc
        except Exception as exc:
            self._abort()
            raise _build_geometry_error("DIGEST_MISMATCH") from exc

    def close(self) -> None:
        if self._state is _CLOSED:
            return
        was_finalized = self._state is _FINALIZED
        had_prior_failure = self._had_failure
        if not was_finalized:
            self._state = _ABORTED
        cleanup_error = self._close_source_sessions()
        spool_error = self._remove_spool_dir()
        first_error = cleanup_error if cleanup_error is not None else spool_error
        self._state = _CLOSED
        if first_error is not None and not had_prior_failure:
            raise _build_geometry_error("CLEANUP_FAILURE") from first_error

    def __enter__(self) -> FrvGeometryLoadSession:
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
    # Pass G1: spatial index spool
    # ------------------------------------------------------------------

    def _make_spool(self) -> None:
        scratch = self._scratch_dir
        try:
            if not isinstance(scratch, Path):
                raise ValueError("scratch_dir must be a pathlib.Path")
            if not scratch.is_dir():
                raise ValueError("scratch_dir must be an existing directory")
            if not os.access(scratch, os.W_OK):
                raise ValueError("scratch_dir must be writable")
            subdir = scratch / ("treeline-geometry-" + uuid.uuid4().hex)
            subdir.mkdir()
        except Exception as exc:  # noqa: BLE001 - spool setup boundary
            raise _build_geometry_error("SCRATCH_STORE") from exc
        self._spool_dir = subdir
        min_xyz = tuple(
            float(value)
            for value in self._source_result.proof.source_bounds_min_xyz
        )
        self._spool = _CellSpool(
            subdir, (min_xyz[0], min_xyz[1], min_xyz[2]), self._provider_config.index_cell_size_m
        )

    def _check_scratch_space(self) -> None:
        required = max(2147483648, 640 * self._source_point_count)
        try:
            free_bytes = shutil.disk_usage(self._scratch_dir).free
        except OSError as exc:  # pragma: no cover - platform boundary
            raise _build_geometry_error("SCRATCH_STORE") from exc
        if free_bytes < required:
            raise _build_geometry_error("SCRATCH_STORE")

    def _run_pass1(self) -> None:
        if not self._scratch_checked:
            self._check_scratch_space()
            self._scratch_checked = True
        self._make_spool()
        assert self._spool is not None
        max_cell_points = self._provider_config.max_index_cell_points
        session1 = load_frv_source(
            self._reference,
            frv_configuration=self._frv_configuration,
            provider_config=self._reconstructed_config,
            chunk_size=self._source_chunk_size,
        )
        self._active_source_sessions.append(session1)
        pass1_expected_index = 0
        try:
            with session1:
                for chunk in session1.iter_chunks():
                    if chunk.source_id != self._expected_source_id:
                        raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
                    if self._pass1_source_id is None:
                        self._pass1_source_id = chunk.source_id
                    elif chunk.source_id != self._pass1_source_id:
                        raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
                    for row in range(chunk.point_count):
                        index = int(chunk.original_index[row])
                        if index < 0:
                            raise _build_geometry_error("INDEX_OUT_OF_RANGE")
                        if index != pass1_expected_index:
                            if index < pass1_expected_index:
                                raise _build_geometry_error("INDEX_DUPLICATE")
                            raise _build_geometry_error("INDEX_LOST")
                        pass1_expected_index += 1
                        x = float(chunk.xyz[row, 0])
                        y = float(chunk.xyz[row, 1])
                        z = float(chunk.xyz[row, 2])
                        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                            raise _build_geometry_error("NONFINITE_XYZ")
                        self._encode_row(index, x, y, z)
                        if self._spool.cell_count(self._spool.cell_key(x, y, z)) > max_cell_points:
                            raise _build_geometry_error("INDEX_CELL_LIMIT")
                result1 = session1.finalize()
            self._check_source_result_matches(result1)
        finally:
            if session1 in self._active_source_sessions:
                self._active_source_sessions.remove(session1)
        if pass1_expected_index != self._source_point_count:
            raise _build_geometry_error("POINT_COUNT_MISMATCH")
        self._spool.close_writes()
        self._state = _INDEX_READY

    def _encode_row(self, index: int, x: float, y: float, z: float) -> None:
        assert self._spool is not None
        try:
            self._spool.append(index, x, y, z)
        except OSError as exc:  # pragma: no cover - IO boundary
            raise _build_geometry_error("SCRATCH_STORE") from exc

    # ------------------------------------------------------------------
    # Pass G2: feature output
    # ------------------------------------------------------------------

    def _run_pass2_source(self) -> None:
        self._pending_chunks = []
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
                        raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
                    if self._pass1_source_id is not None and source_chunk.source_id != (
                        self._pass1_source_id
                    ):
                        raise _build_geometry_error("SOURCE_IDENTITY_MISMATCH")
                    data = self._build_geometry_chunk_data(source_chunk)
                    self._pending_chunks.append(data)
                result2 = session2.finalize()
            self._check_source_result_matches(result2)
            self._result2 = result2
        finally:
            if session2 in self._active_source_sessions:
                self._active_source_sessions.remove(session2)

    def _build_geometry_chunk_data(self, source_chunk: Any) -> tuple:
        point_count = source_chunk.point_count
        scalars = {
            "source_id": source_chunk.source_id,
            "chunk_id": source_chunk.chunk_id,
            "start_index": source_chunk.start_index,
            "point_count": point_count,
            "is_last": source_chunk.is_last,
        }
        arrays: dict[str, np.ndarray] = {
            "original_index": np.empty(point_count, dtype="<i8"),
            "raw_neighbor_count": np.empty((point_count, 3), dtype="<u4"),
            "neighbor_count": np.empty((point_count, 3), dtype="<u2"),
            "density_per_m3": np.empty((point_count, 3), dtype="<f4"),
            "local_spacing_m": np.empty((point_count, 3), dtype="<f4"),
            "eigenvalues_m2": np.empty((point_count, 3, 3), dtype="<f8"),
            "principal_direction": np.empty((point_count, 3, 3), dtype="<f4"),
            "surface_normal": np.empty((point_count, 3, 3), dtype="<f4"),
            "linearity": np.empty((point_count, 3), dtype="<f4"),
            "planarity": np.empty((point_count, 3), dtype="<f4"),
            "scattering": np.empty((point_count, 3), dtype="<f4"),
            "verticality": np.empty((point_count, 3), dtype="<f4"),
            "vertical_span_m": np.empty(point_count, dtype="<f4"),
            "vertical_occupied_slices": np.empty(point_count, dtype="<u2"),
            "above_count": np.empty(point_count, dtype="<u2"),
            "below_count": np.empty(point_count, dtype="<u2"),
            "support_neighbor_count": np.empty(point_count, dtype="|u1"),
            "support_neighbor_original_index": np.empty((point_count, 16), dtype="<i8"),
            "geometry_evidence_flags": np.empty(point_count, dtype="<u4"),
            "geometry_evidence_state": np.empty(point_count, dtype="|u1"),
            "reason_flags": np.empty(point_count, dtype="<u4"),
            "quality_flags": np.empty(point_count, dtype="<u4"),
        }
        support_arrays = {
            "support_neighbor_count": arrays["support_neighbor_count"],
            "support_neighbor_original_index": arrays["support_neighbor_original_index"],
        }
        for row in range(point_count):
            point = self._compute_point_geometry(source_chunk, row)
            arrays["original_index"][row] = point["original_index"]
            for scale in range(_SCALE_COUNT):
                arrays["raw_neighbor_count"][row, scale] = point["raw_neighbor_count"][scale]
                arrays["neighbor_count"][row, scale] = point["neighbor_count"][scale]
                arrays["density_per_m3"][row, scale] = point["density_per_m3"][scale]
                arrays["local_spacing_m"][row, scale] = point["local_spacing_m"][scale]
                for axis in range(3):
                    arrays["eigenvalues_m2"][row, scale, axis] = point[
                        "eigenvalues_m2"
                    ][scale][axis]
                    arrays["principal_direction"][row, scale, axis] = point[
                        "principal_direction"
                    ][scale][axis]
                    arrays["surface_normal"][row, scale, axis] = point[
                        "surface_normal"
                    ][scale][axis]
                arrays["linearity"][row, scale] = point["linearity"][scale]
                arrays["planarity"][row, scale] = point["planarity"][scale]
                arrays["scattering"][row, scale] = point["scattering"][scale]
                arrays["verticality"][row, scale] = point["verticality"][scale]
            arrays["vertical_span_m"][row] = point["vertical_span_m"]
            arrays["vertical_occupied_slices"][row] = point["vertical_occupied_slices"]
            arrays["above_count"][row] = point["above_count"]
            arrays["below_count"][row] = point["below_count"]
            arrays["geometry_evidence_flags"][row] = point["geometry_evidence_flags"]
            arrays["geometry_evidence_state"][row] = point["geometry_evidence_state"]
            arrays["reason_flags"][row] = point["reason_flags"]
            arrays["quality_flags"][row] = point["quality_flags"]
            self._support_candidates.append(point["support_candidates"])
            self._support_sets.append(point["support_candidate_set"])
            self._support_valid.append(point["support_valid"])
            self._support_canopy.append(point["support_canopy"])
            self._support_direction.append(point["support_direction"])
        # Support arrays are filled during edge resolution, not here.
        return (scalars, arrays, support_arrays)

    def _resolve_support_edges(self) -> None:
        cosine_min = self._provider_config.direction_cosine_min
        for index in range(len(self._support_candidates)):
            candidates = self._support_candidates[index]
            if not candidates:
                continue
            i_valid = self._support_valid[index]
            i_canopy = self._support_canopy[index]
            i_direction = self._support_direction[index]
            edges = []
            for d2, neighbor in candidates:
                if index not in self._support_sets[neighbor]:
                    continue
                if not (i_valid and self._support_valid[neighbor]):
                    continue
                dot = (
                    i_direction[0] * self._support_direction[neighbor][0]
                    + i_direction[1] * self._support_direction[neighbor][1]
                    + i_direction[2] * self._support_direction[neighbor][2]
                )
                if abs(dot) < cosine_min and not (i_canopy and self._support_canopy[neighbor]):
                    continue
                edges.append((d2, neighbor))
            self._support_candidates[index] = edges
            self._support_sets[index] = frozenset(neighbor for _d2, neighbor in edges)

    def _assemble_geometry_chunk(self, data: tuple) -> FrvGeometryChunk:
        scalars, arrays, support_arrays = data
        point_count = scalars["point_count"]
        arrays = dict(arrays)
        resolved_support_count = np.empty(point_count, dtype="|u1")
        resolved_support_index = np.empty((point_count, 16), dtype="<i8")
        for row in range(point_count):
            global_index = scalars["start_index"] + row
            edges = self._support_candidates[global_index]
            support_count = len(edges)
            resolved_support_count[row] = support_count
            for slot in range(16):
                if slot < support_count:
                    resolved_support_index[row, slot] = edges[slot][1]
                else:
                    resolved_support_index[row, slot] = -1
        arrays["support_neighbor_count"] = resolved_support_count
        arrays["support_neighbor_original_index"] = resolved_support_index
        frozen = {}
        for name, array in arrays.items():
            frozen[name] = np.ascontiguousarray(array)
            frozen[name].setflags(write=False)
        return FrvGeometryChunk(
            source_id=scalars["source_id"],
            chunk_id=scalars["chunk_id"],
            start_index=scalars["start_index"],
            point_count=point_count,
            original_index=frozen["original_index"],
            raw_neighbor_count=frozen["raw_neighbor_count"],
            neighbor_count=frozen["neighbor_count"],
            density_per_m3=frozen["density_per_m3"],
            local_spacing_m=frozen["local_spacing_m"],
            eigenvalues_m2=frozen["eigenvalues_m2"],
            principal_direction=frozen["principal_direction"],
            surface_normal=frozen["surface_normal"],
            linearity=frozen["linearity"],
            planarity=frozen["planarity"],
            scattering=frozen["scattering"],
            verticality=frozen["verticality"],
            vertical_span_m=frozen["vertical_span_m"],
            vertical_occupied_slices=frozen["vertical_occupied_slices"],
            above_count=frozen["above_count"],
            below_count=frozen["below_count"],
            support_neighbor_count=frozen["support_neighbor_count"],
            support_neighbor_original_index=frozen["support_neighbor_original_index"],
            geometry_evidence_flags=frozen["geometry_evidence_flags"],
            geometry_evidence_state=frozen["geometry_evidence_state"],
            reason_flags=frozen["reason_flags"],
            quality_flags=frozen["quality_flags"],
            is_last=scalars["is_last"],
        )

    # ------------------------------------------------------------------
    # Per-point multiscale geometry computation
    # ------------------------------------------------------------------

    def _compute_point_geometry(self, source_chunk: Any, row: int) -> dict[str, Any]:
        i = int(source_chunk.original_index[row])
        x = float(source_chunk.xyz[row, 0])
        y = float(source_chunk.xyz[row, 1])
        z = float(source_chunk.xyz[row, 2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise _build_geometry_error("NONFINITE_XYZ")
        assert self._spool is not None
        config = self._provider_config
        radii = config.radii_m
        radius_squared = [r * r for r in radii]
        support_radius_sq = config.support_neighbor_radius_m * config.support_neighbor_radius_m
        vertical_radius_sq = (
            config.vertical_cylinder_radius_m * config.vertical_cylinder_radius_m
        )
        vertical_half = config.vertical_half_window_m
        halo = config.radii_m[_SCALE_LARGE]

        scale_heaps: list[list[tuple[Any, ...]]] = [[], [], []]
        raw_counts = [0, 0, 0]
        support_heap: list[tuple[Any, ...]] = []
        support_raw = 0
        span_min = z
        span_max = z
        above = 0
        below = 0
        slice_bitmap = 0

        for cell_key in self._halo_cell_keys(x, y, z, halo):
            for j, xj, yj, zj in self._spool.read_cell(cell_key):
                if j == i:
                    continue
                dx = xj - x
                dy = yj - y
                dz = zj - z
                d2 = dx * dx + dy * dy + dz * dz
                if not math.isfinite(d2):
                    continue
                for scale in range(_SCALE_COUNT):
                    if d2 <= radius_squared[scale]:
                        raw_counts[scale] += 1
                        _heap_push(
                            scale_heaps[scale], (d2, j), (xj, yj, zj),
                            config.max_neighbors_per_scale,
                        )
                if d2 <= support_radius_sq:
                    support_raw += 1
                    _heap_push(support_heap, (d2, j), (), config.support_neighbor_cap)
                radial_sq = d2 - dz * dz
                if radial_sq <= vertical_radius_sq and abs(dz) <= vertical_half:
                    if dz > 0.05:
                        above += 1
                    elif dz < -0.05:
                        below += 1
                    slice_id = math.floor((dz + vertical_half) / config.vertical_slice_height_m)
                    if slice_id > 11:
                        slice_id = 11
                    slice_bitmap |= 1 << slice_id
                    if zj < span_min:
                        span_min = zj
                    if zj > span_max:
                        span_max = zj

        eigenvalues = []
        directions = []
        normals = []
        linearity = []
        planarity = []
        scattering = []
        verticality = []
        densities = []
        spacings = []
        neighbor_counts = []
        medium_direction = (0.0, 0.0, 0.0)
        for scale in range(_SCALE_COUNT):
            neighbors = _heap_items(scale_heaps[scale])
            raw = raw_counts[scale]
            count = min(raw, config.max_neighbors_per_scale)
            neighbor_counts.append(count)
            radius = float(radii[scale])
            volume = (4.0 / 3.0) * math.pi * radius**3
            density = raw / volume
            densities.append(_normalize_zero(density))
            if count == 0:
                spacing = 0.0
            else:
                distances = np.asarray(
                    [math.sqrt(item[0]) for item in neighbors], dtype=np.float64
                )
                spacing = float(np.median(distances))
            spacings.append(_normalize_zero(spacing))
            if count >= int(config.min_neighbor_counts[scale]):
                eigen, direction, normal, feat_linear, feat_planar, feat_scatter, feat_vert = (
                    self._scale_pca(i, x, y, z, neighbors)
                )
                eigenvalues.append(eigen)
                directions.append(direction)
                normals.append(normal)
                linearity.append(feat_linear)
                planarity.append(feat_planar)
                scattering.append(feat_scatter)
                verticality.append(feat_vert)
                if scale == _SCALE_MEDIUM:
                    medium_direction = direction
            else:
                eigenvalues.append((0.0, 0.0, 0.0))
                directions.append((0.0, 0.0, 0.0))
                normals.append((0.0, 0.0, 0.0))
                linearity.append(0.0)
                planarity.append(0.0)
                scattering.append(0.0)
                verticality.append(0.0)

        support_candidates = _heap_items(support_heap)
        support_candidate_set = frozenset(neighbor for _d2, neighbor in support_candidates)

        state, reason = self._determine_state(neighbor_counts, eigenvalues)
        vertical_span = span_max - span_min
        occupied_slices = bin(slice_bitmap).count("1")
        flags = self._compute_evidence_flags(
            state, densities, eigenvalues, linearity, planarity, scattering, verticality,
            directions, normals, vertical_span, occupied_slices, above, below,
        )
        quality = 0
        if any(raw > config.max_neighbors_per_scale for raw in raw_counts):
            quality |= int(FrvGeometryQualityFlag.NEIGHBOR_CAP_REACHED)
        if support_raw > config.support_neighbor_cap:
            quality |= int(FrvGeometryQualityFlag.SUPPORT_NEIGHBOR_CAP_REACHED)

        return {
            "original_index": i,
            "raw_neighbor_count": tuple(raw_counts),
            "neighbor_count": tuple(neighbor_counts),
            "density_per_m3": tuple(np.float32(value) for value in densities),
            "local_spacing_m": tuple(np.float32(value) for value in spacings),
            "eigenvalues_m2": tuple(eigenvalues),
            "principal_direction": tuple(directions),
            "surface_normal": tuple(normals),
            "linearity": tuple(np.float32(value) for value in linearity),
            "planarity": tuple(np.float32(value) for value in planarity),
            "scattering": tuple(np.float32(value) for value in scattering),
            "verticality": tuple(np.float32(value) for value in verticality),
            "vertical_span_m": np.float32(_normalize_zero(vertical_span)),
            "vertical_occupied_slices": occupied_slices,
            "above_count": above,
            "below_count": below,
            "geometry_evidence_flags": flags,
            "geometry_evidence_state": state,
            "reason_flags": reason,
            "quality_flags": quality,
            "support_candidates": support_candidates,
            "support_candidate_set": support_candidate_set,
            "support_valid": state == int(FrvGeometryEvidenceState.VALID),
            "support_canopy": bool(
                flags & int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)
            ),
            "support_direction": medium_direction,
        }

    def _halo_cell_keys(
        self, x: float, y: float, z: float, radius: float
    ) -> list[tuple[int, int, int]]:
        assert self._spool is not None
        cell_size = self._provider_config.index_cell_size_m
        min_x = float(self._source_result.proof.source_bounds_min_xyz[0])
        min_y = float(self._source_result.proof.source_bounds_min_xyz[1])
        min_z = float(self._source_result.proof.source_bounds_min_xyz[2])
        ix_lo = math.floor((x - radius - min_x) / cell_size)
        ix_hi = math.floor((x + radius - min_x) / cell_size)
        iy_lo = math.floor((y - radius - min_y) / cell_size)
        iy_hi = math.floor((y + radius - min_y) / cell_size)
        iz_lo = math.floor((z - radius - min_z) / cell_size)
        iz_hi = math.floor((z + radius - min_z) / cell_size)
        keys = []
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                for iz in range(iz_lo, iz_hi + 1):
                    keys.append((ix, iy, iz))
        return keys

    def _scale_pca(self, i: int, x: float, y: float, z: float, neighbors):
        """Covariance/PCA for one scale with frozen sign canonicalization."""

        samples = [(i, x, y, z)] + [
            (j, xj, yj, zj) for _d2, j, xj, yj, zj in neighbors
        ]
        samples.sort(key=lambda sample: sample[0])
        count = len(samples)
        mean = [0.0, 0.0, 0.0]
        for _sample_index, sx, sy, sz in samples:
            mean[0] += sx
            mean[1] += sy
            mean[2] += sz
        mean = [value / count for value in mean]
        covariance = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        for _sample_index, sx, sy, sz in samples:
            dx = sx - mean[0]
            dy = sy - mean[1]
            dz = sz - mean[2]
            covariance[0][0] += dx * dx
            covariance[0][1] += dx * dy
            covariance[0][2] += dx * dz
            covariance[1][0] += dy * dx
            covariance[1][1] += dy * dy
            covariance[1][2] += dy * dz
            covariance[2][0] += dz * dx
            covariance[2][1] += dz * dy
            covariance[2][2] += dz * dz
        for axis_i in range(3):
            for axis_j in range(3):
                covariance[axis_i][axis_j] /= count
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance, dtype=np.float64))
        except Exception as exc:  # noqa: BLE001 - eigensystem failure boundary
            raise _build_geometry_error("PCA_INVALID") from exc
        if not (
            np.isfinite(eigenvalues).all()
            and np.isfinite(eigenvectors).all()
        ):
            raise _build_geometry_error("PCA_INVALID")
        floor = self._provider_config.eigenvalue_floor_m2
        for axis in range(3):
            value = float(eigenvalues[axis])
            if value < -floor:
                raise _build_geometry_error("EIGENVALUE_INVALID")
            if value < 0.0:
                eigenvalues[axis] = 0.0
        eigen = tuple(_normalize_zero(float(value)) for value in eigenvalues)
        floor = self._provider_config.eigenvalue_floor_m2
        if eigen[2] <= floor:
            return (eigen, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0)
        linearity = (eigen[2] - eigen[1]) / eigen[2]
        planarity = (eigen[1] - eigen[0]) / eigen[2]
        scattering = eigen[0] / eigen[2]
        if eigen[2] - eigen[1] > floor:
            direction = _sign_canonicalized(eigenvectors[:, 2])
            verticality = 1.0 - abs(direction[2])
        else:
            direction = (0.0, 0.0, 0.0)
            verticality = 1.0
        if eigen[1] - eigen[0] > floor:
            normal = _sign_canonicalized(eigenvectors[:, 0])
        else:
            normal = (0.0, 0.0, 0.0)
        direction = tuple(_normalize_zero(component) for component in direction)
        normal = tuple(_normalize_zero(component) for component in normal)
        return (
            eigen,
            direction,
            normal,
            _normalize_zero(linearity),
            _normalize_zero(planarity),
            _normalize_zero(scattering),
            _normalize_zero(verticality),
        )

    def _determine_state(
        self,
        neighbor_counts: list[int],
        eigenvalues: list[tuple[float, float, float]],
    ) -> tuple[int, int]:
        config = self._provider_config
        medium_at = neighbor_counts[_SCALE_MEDIUM] >= int(config.min_neighbor_counts[_SCALE_MEDIUM])
        large_at = neighbor_counts[_SCALE_LARGE] >= int(config.min_neighbor_counts[_SCALE_LARGE])
        if not (medium_at and large_at):
            return int(FrvGeometryEvidenceState.INSUFFICIENT), int(
                FrvGeometryReasonFlag.INSUFFICIENT_NEIGHBORS
            )
        floor = config.eigenvalue_floor_m2
        if eigenvalues[_SCALE_MEDIUM][2] <= floor or eigenvalues[_SCALE_LARGE][2] <= floor:
            return int(FrvGeometryEvidenceState.DEGENERATE), int(
                FrvGeometryReasonFlag.DEGENERATE_COVARIANCE
            )
        return int(FrvGeometryEvidenceState.VALID), int(FrvGeometryReasonFlag.NONE)

    def _compute_evidence_flags(
        self,
        state: int,
        densities: list[float],
        eigenvalues: list[tuple[float, float, float]],
        linearity: list[float],
        planarity: list[float],
        scattering: list[float],
        verticality: list[float],
        directions: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]],
        vertical_span: float,
        occupied_slices: int,
        above: int,
        below: int,
    ) -> int:
        config = self._provider_config
        valid = state == int(FrvGeometryEvidenceState.VALID)
        flags = 0
        density_supported = (
            densities[_SCALE_LARGE] >= config.canopy_density_min_per_m3
        )
        vertical_continuity = (
            vertical_span >= config.woody_vertical_span_min_m
            and occupied_slices >= config.min_vertical_occupied_slices
            and above >= 1
            and below >= 1
        )
        if valid:
            flags |= int(FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT)
        if density_supported:
            flags |= int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED)
        if valid and density_supported:
            if (
                scattering[_SCALE_LARGE] >= config.canopy_scattering_min
                and linearity[_SCALE_LARGE] <= config.canopy_linearity_max
            ):
                flags |= int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE)
        if vertical_continuity:
            flags |= int(FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY)
        if valid and vertical_continuity:
            if (
                linearity[_SCALE_MEDIUM] >= config.woody_linearity_min
                and verticality[_SCALE_MEDIUM] <= config.woody_verticality_max
            ):
                flags |= int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE)
        if valid:
            if (
                planarity[_SCALE_MEDIUM] >= config.ground_planarity_min
                and abs(normals[_SCALE_MEDIUM][2]) >= config.ground_normal_z_min
            ):
                flags |= int(FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE)
        return flags

    # ------------------------------------------------------------------
    # Digest feed, continuity, counts, and proof
    # ------------------------------------------------------------------

    def _check_geometry_stream_continuity(self, chunks: list[FrvGeometryChunk]) -> None:
        expected = 0
        for chunk in chunks:
            for row in range(chunk.point_count):
                index = int(chunk.original_index[row])
                if index < 0:
                    raise _build_geometry_error("INDEX_OUT_OF_RANGE")
                if index != expected:
                    if index < expected:
                        raise _build_geometry_error("INDEX_DUPLICATE")
                    raise _build_geometry_error("INDEX_LOST")
                expected += 1
        if expected != self._source_point_count:
            raise _build_geometry_error("SOURCE_PASS_INCOMPLETE")

    def _accumulate_counts(self, chunk: FrvGeometryChunk) -> None:
        for row in range(chunk.point_count):
            state = int(chunk.geometry_evidence_state[row])
            if state == int(FrvGeometryEvidenceState.VALID):
                self._valid_count += 1
            elif state == int(FrvGeometryEvidenceState.INSUFFICIENT):
                self._insufficient_count += 1
            else:
                self._degenerate_count += 1
            flags = int(chunk.geometry_evidence_flags[row])
            if flags & int(FrvGeometryEvidenceFlag.NEIGHBORHOOD_SUFFICIENT):
                self._neighborhood_sufficient_count += 1
            if flags & int(FrvGeometryEvidenceFlag.DENSITY_SUPPORTED):
                self._density_supported_count += 1
            if flags & int(FrvGeometryEvidenceFlag.CANOPY_BRANCHING_COMPATIBLE):
                self._canopy_branching_compatible_count += 1
            if flags & int(FrvGeometryEvidenceFlag.WOODY_COLUMN_COMPATIBLE):
                self._woody_column_compatible_count += 1
            if flags & int(FrvGeometryEvidenceFlag.GROUND_SURFACE_COMPATIBLE):
                self._ground_surface_compatible_count += 1
            if flags & int(FrvGeometryEvidenceFlag.VERTICAL_CONTINUITY):
                self._vertical_continuity_count += 1
            quality = int(chunk.quality_flags[row])
            if quality & int(FrvGeometryQualityFlag.NEIGHBOR_CAP_REACHED):
                self._neighbor_cap_reached_count += 1
            if quality & int(FrvGeometryQualityFlag.SUPPORT_NEIGHBOR_CAP_REACHED):
                self._support_neighbor_cap_reached_count += 1

    def _check_source_result_matches(self, result: FrvSourceProviderResult) -> None:
        expected = self._source_result
        if result.proof.to_dict() != expected.proof.to_dict():
            raise _build_geometry_error("DIGEST_MISMATCH")
        if result.source_identity != expected.source_identity:
            raise _build_geometry_error("DIGEST_MISMATCH")
        if (
            to_frv_a2_provider_binding_evidence(result).binding_digest
            != to_frv_a2_provider_binding_evidence(expected).binding_digest
        ):
            raise _build_geometry_error("DIGEST_MISMATCH")

    def _build_proof(self) -> FrvGeometryProviderProof:
        assert self._result2 is not None
        source_proof = self._source_result.proof
        source_proof_digest = frv_digest(
            {
                "digest_domain": _FRV_GEOMETRY_SOURCE_PROOF_DIGEST_DOMAIN,
                "source_proof": source_proof.to_dict(),
            }
        )
        source_binding_digest = to_frv_a2_provider_binding_evidence(
            self._source_result
        ).binding_digest
        components = source_proof.artifact_components
        artifact_reference_id = _derive_artifact_reference_id(
            FrvA2ProviderRole.GEOMETRY, FRV_GEOMETRY_PROVIDER_KEY, components
        )
        proof = FrvGeometryProviderProof.__new__(FrvGeometryProviderProof)
        object.__setattr__(proof, "schema_version", FRV_GEOMETRY_PROOF_SCHEMA_VERSION)
        object.__setattr__(proof, "provider_key", FRV_GEOMETRY_PROVIDER_KEY)
        object.__setattr__(proof, "provider_role", FrvA2ProviderRole.GEOMETRY)
        object.__setattr__(
            proof, "interface_version", FRV_GEOMETRY_PROVIDER_INTERFACE_VERSION
        )
        object.__setattr__(proof, "algorithm_version", FRV_GEOMETRY_ALGORITHM_VERSION)
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
        assert self._geometry_point_content_digest is not None
        object.__setattr__(
            proof,
            "geometry_point_content_digest",
            self._geometry_point_content_digest,
        )
        object.__setattr__(proof, "valid_count", self._valid_count)
        object.__setattr__(proof, "insufficient_count", self._insufficient_count)
        object.__setattr__(proof, "degenerate_count", self._degenerate_count)
        object.__setattr__(
            proof,
            "neighborhood_sufficient_count",
            self._neighborhood_sufficient_count,
        )
        object.__setattr__(
            proof, "density_supported_count", self._density_supported_count
        )
        object.__setattr__(
            proof,
            "canopy_branching_compatible_count",
            self._canopy_branching_compatible_count,
        )
        object.__setattr__(
            proof, "woody_column_compatible_count", self._woody_column_compatible_count
        )
        object.__setattr__(
            proof,
            "ground_surface_compatible_count",
            self._ground_surface_compatible_count,
        )
        object.__setattr__(
            proof, "vertical_continuity_count", self._vertical_continuity_count
        )
        object.__setattr__(
            proof, "neighbor_cap_reached_count", self._neighbor_cap_reached_count
        )
        object.__setattr__(
            proof,
            "support_neighbor_cap_reached_count",
            self._support_neighbor_cap_reached_count,
        )
        object.__setattr__(proof, "coordinate_frame", "LOCAL_CARTESIAN_XYZ")
        object.__setattr__(proof, "units", "m")
        object.__setattr__(proof, "origin_xyz", source_proof.origin_xyz)
        object.__setattr__(proof, "transform_declaration", "identity")
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
        self._remove_spool_dir()
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

    def _remove_spool_dir(self) -> BaseException | None:
        if self._spool_dir is None:
            return None
        if self._spool is not None:
            try:
                self._spool.close_writes()
            except Exception:  # noqa: BLE001 - cleanup path
                pass
        try:
            shutil.rmtree(self._spool_dir)
        except OSError as exc:  # pragma: no cover - platform boundary
            return exc
        finally:
            self._spool_dir = None
        return None


def load_frv_geometry(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvGeometryProviderConfig,
    scratch_dir: Path,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvGeometryLoadSession:
    """Open a single-use two-pass P-GEOMETRY session (private implementation)."""

    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(frv_configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")
    if type(provider_config) is not FrvGeometryProviderConfig:
        raise TypeError("provider_config must be a FrvGeometryProviderConfig")
    if not isinstance(scratch_dir, Path):
        raise TypeError("scratch_dir must be a pathlib.Path")
    if type(source_chunk_size) is not int:
        raise TypeError("source_chunk_size must be a built-in int")
    if source_chunk_size <= 0:
        raise ValueError("source_chunk_size must be positive")
    return _FrvGeometryLoadSession(
        reference,
        source_result,
        frv_configuration,
        provider_config,
        scratch_dir,
        source_chunk_size,
    )
