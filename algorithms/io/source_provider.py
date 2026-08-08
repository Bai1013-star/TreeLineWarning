"""FRV-A2 P-SOURCE runtime provider session (C4/C5).

Implements the frozen ``FrvSourceLoadSession`` protocol over a single laspy
handle.  Guarantees: no full-point materialization, no repeated point reads,
no LAZ/compressed/COPC input, single-use iterator, TOCTOU fail-closed
finalization, and success-only BND1 evidence via the C3 finalized factory.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from types import MappingProxyType
from typing import Any, Literal

import laspy
import numpy as np
from core.contracts.frv import (
    FrvConfiguration,
    FrvErrorCode,
    FrvQualityFlag,
    FrvReasonFlag,
    canonical_frv_json_bytes,
)
from core.contracts.frv_a2 import (
    FrvA2DomainKind,
    FrvA2FailurePhase,
    FrvA2FailureRecord,
    FrvA2ProviderRole,
)
from core.contracts.frv_source import (
    _FRV_SOURCE_ATTRIBUTE_DTYPES,
    _FRV_SOURCE_FINALIZED_CAPABILITY,
    FRV_SOURCE_DEFAULT_CHUNK_SIZE,
    FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
    FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION,
    FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
    FRV_SOURCE_PROVIDER_KEY,
    FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
    FrvSourceChunk,
    FrvSourceCompressionState,
    FrvSourceCrsAuthorityState,
    FrvSourceFormat,
    FrvSourceHeaderMetadata,
    FrvSourceLoadError,
    FrvSourceLoadSession,
    FrvSourceProviderConfig,
    FrvSourceProviderProof,
    FrvSourceProviderReference,
    FrvSourceProviderResult,
    FrvSourceVlrPolicyState,
    _create_finalized_frv_source_provider_result,
    validate_frv_source_provider_reference,
)

# Private session-state markers (never exposed; no public `.state`).
_CREATED = "CREATED"
_ITERATING = "ITERATING"
_STREAM_COMPLETE = "STREAM_COMPLETE"
_FINALIZED = "FINALIZED"
_ABORTED = "ABORTED"
_CLOSED = "CLOSED"

# Frozen plan 15.4 mapping keys -> failure-record tuple.  Only existing A2
# enums/flags are used; this is the implementation of the single frozen table.
_FRV_SOURCE_FAILURE_MAP: dict[
    str, tuple[FrvA2FailurePhase, str, FrvErrorCode, FrvQualityFlag]
] = {
    "FILE_MISSING": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "runtime_access_path",
        FrvErrorCode.FRV_PROVIDER_MISSING,
        FrvQualityFlag.PROVIDER_MISSING,
    ),
    "REFERENCE_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "reference",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "FORMAT_REJECTED": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "source_format",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "ZERO_POINT_SOURCE": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "source_point_count",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "HEADER_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "header",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "SOURCE_IDENTITY_MISMATCH": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "source_identity",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "COORDINATE_AUTHORITY_REJECTED": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "coordinate_frame",
        FrvErrorCode.FRV_SOURCE_IDENTITY_FAILED,
        FrvQualityFlag.SOURCE_IDENTITY_FAILURE,
    ),
    "NONFINITE_COORDINATE": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "xyz",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "SESSION_ABORTED": (
        FrvA2FailurePhase.DOMAIN_ALIGNMENT,
        "session_state",
        FrvErrorCode.FRV_STREAM_INCOMPLETE,
        FrvQualityFlag.STREAM_INCOMPLETE,
    ),
    "TOCTOU_CHANGED": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "source_file",
        FrvErrorCode.FRV_ARTIFACT_DIGEST_MISMATCH,
        FrvQualityFlag.ARTIFACT_DIGEST_MISMATCH,
    ),
    "FINALIZE_INVALID_STATE": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "session_state",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
}

_KNOWN_CRS_VLR_TYPES = frozenset(
    {
        "GeoKeyDirectoryVlr",
        "GeoDoubleParamsVlr",
        "GeoAsciiParamsVlr",
        "WktCoordinateSystemVlr",
        "WktMathTransformVlr",
    }
)


def _build_load_error(key: str) -> FrvSourceLoadError:
    phase, field_name, error_code, quality_flag = _FRV_SOURCE_FAILURE_MAP[key]
    reason_flags = int(FrvReasonFlag.PROVIDER_UNCERTAINTY)
    detail = (
        f"{phase.value}:{FrvA2ProviderRole.SOURCE_SCAN.value}:"
        f"{FRV_SOURCE_PROVIDER_KEY}:{field_name}:{error_code.value}:{reason_flags}"
    )
    record = FrvA2FailureRecord(
        phase,
        FrvA2ProviderRole.SOURCE_SCAN,
        FRV_SOURCE_PROVIDER_KEY,
        field_name,
        error_code,
        reason_flags,
        detail,
    )
    return FrvSourceLoadError(failure_record=record, quality_flags=int(quality_flag))


def _stream_index_domain_digest(total: int, source_content_identity_digest: str) -> str:
    """Return the FRVA2IDX1 digest over the global 0..N-1 domain, in-stream.

    Byte-identical to ``frv_a2_index_domain_digest`` for
    ``tuple(range(total))``, but hashes the lazy ``range`` and never builds an
    N-length index collection.
    """

    hasher = hashlib.sha256()
    hasher.update(b"FRVA2IDX1\0")
    hasher.update(b"ORIGINAL_INDEX")
    hasher.update(b"\0")
    hasher.update(source_content_identity_digest.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(FrvA2DomainKind.FULL_SOURCE.value.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(struct.pack("<Q", total))
    for index in range(total):
        hasher.update(struct.pack("<Q", index))
    return hasher.hexdigest()


class _FrvSourceLoadSession:
    """Single-use P-SOURCE load session implementing the C3 protocol."""

    def __init__(
        self,
        reference: FrvSourceProviderReference,
        frv_configuration: FrvConfiguration,
        provider_config: FrvSourceProviderConfig,
        chunk_size: int,
    ) -> None:
        if type(reference) is not FrvSourceProviderReference:
            raise TypeError("reference must be a FrvSourceProviderReference")
        if type(frv_configuration) is not FrvConfiguration:
            raise TypeError("frv_configuration must be a FrvConfiguration")
        if type(provider_config) is not FrvSourceProviderConfig:
            raise TypeError("provider_config must be a FrvSourceProviderConfig")
        if type(chunk_size) is not int:
            raise TypeError("chunk_size must be a built-in int")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self._reference = reference
        self._frv_configuration = frv_configuration
        self._provider_config = provider_config
        self._chunk_size = chunk_size
        self._state = _CREATED
        self._had_failure = False
        self._result: FrvSourceProviderResult | None = None
        self._handle: Any = None
        self._reader: Any = None
        self._initial_sha: str | None = None
        self._initial_size: int | None = None
        self._total_points: int | None = None
        self._streamed_bounds: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        ) = None

        # In-stream FRVPCD1 accumulator (byte-identical to the authority
        # framing; required by the single-pass streaming contract).
        self._canonical_spec = reference.canonical_point_content_spec
        self._frvpcd1: Any = None
        if self._canonical_spec is not None:
            self._frvpcd1 = hashlib.sha256()
            self._frvpcd1.update(b"FRVPCD1\0")
            spec_header = canonical_frv_json_bytes(self._canonical_spec.to_digest_dict())
            self._frvpcd1.update(struct.pack("<Q", len(spec_header)))
            self._frvpcd1.update(spec_header)

        try:
            if reference.runtime_access_path.suffix.lower() != ".las":
                raise _build_load_error("FORMAT_REJECTED")
            self._handle = open(reference.runtime_access_path, "rb")
            self._handle.seek(0)
            self._initial_sha, self._initial_size = self._hash_handle(self._handle)
            self._handle.seek(0)
            self._reader = laspy.open(self._handle, closefd=False, read_evlrs=True)
            self._header_metadata, self._las_header = self._read_and_validate_header(
                self._reader.header
            )
        except FrvSourceLoadError:
            self._abort()
            raise
        except FileNotFoundError as exc:
            self._abort()
            raise _build_load_error("FILE_MISSING") from exc
        except Exception as exc:
            self._abort()
            raise _build_load_error("REFERENCE_INVALID") from exc

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[FrvSourceChunk]:
        if self._state is not _CREATED:
            raise _build_load_error("FINALIZE_INVALID_STATE")
        self._state = _ITERATING
        return self._stream_chunks()

    def _stream_chunks(self):
        try:
            raw_iterator = self._reader.chunk_iterator(self._chunk_size)
            pending = None
            pending_count = 0
            start_index = 0
            chunk_id = 0
            bounds_min: Any = None
            bounds_max: Any = None
            for raw in raw_iterator:
                if pending is not None:
                    chunk = self._build_chunk(
                        pending, pending_count, start_index, chunk_id, is_last=False
                    )
                    start_index += pending_count
                    chunk_id += 1
                    bounds_min, bounds_max = self._accumulate_bounds(
                        chunk, bounds_min, bounds_max
                    )
                    self._feed_frvpcd1(chunk)
                    yield chunk
                pending = raw
                pending_count = len(raw.array)
            if pending is not None:
                chunk = self._build_chunk(
                    pending, pending_count, start_index, chunk_id, is_last=True
                )
                start_index += pending_count
                bounds_min, bounds_max = self._accumulate_bounds(
                    chunk, bounds_min, bounds_max
                )
                self._feed_frvpcd1(chunk)
                yield chunk
            self._total_points = start_index
            self._streamed_bounds = self._freeze_bounds(bounds_min, bounds_max)
            if self._total_points != self._header_metadata.point_count:
                raise _build_load_error("SESSION_ABORTED")
            reader_error = self._close_reader()
            if reader_error is not None:
                raise _build_load_error("SESSION_ABORTED") from reader_error
            self._state = _STREAM_COMPLETE
        except GeneratorExit:
            self._mark_aborted()
            self._release_resources()
            raise
        except FrvSourceLoadError:
            self._abort()
            raise
        except Exception as exc:
            self._abort()
            raise _build_load_error("SESSION_ABORTED") from exc

    def finalize(self) -> FrvSourceProviderResult:
        if self._state in (_CREATED, _ITERATING, _ABORTED):
            raise _build_load_error("SESSION_ABORTED")
        if self._state is not _STREAM_COMPLETE:
            raise _build_load_error("FINALIZE_INVALID_STATE")
        try:
            try:
                assert self._handle is not None
                self._handle.seek(0)
                final_sha, final_size = self._hash_handle(self._handle)
            except FrvSourceLoadError:
                raise
            except Exception as exc:
                raise _build_load_error("REFERENCE_INVALID") from exc
            self._check_toctou(final_sha, final_size)
            proof = self._build_proof()
            result = _create_finalized_frv_source_provider_result(
                proof,
                self._reference.source_identity,
                capability=_FRV_SOURCE_FINALIZED_CAPABILITY,
            )
            self._result = result
            handle_error = self._close_handle()
            if handle_error is not None:
                raise _build_load_error("FINALIZE_INVALID_STATE") from handle_error
            self._state = _FINALIZED
            return result
        except FrvSourceLoadError:
            self._abort()
            raise
        except ValueError as exc:
            self._abort()
            raise _build_load_error("SOURCE_IDENTITY_MISMATCH") from exc
        except Exception as exc:
            self._abort()
            raise _build_load_error("FINALIZE_INVALID_STATE") from exc

    def close(self) -> None:
        if self._state is _CLOSED:
            return
        was_finalized = self._state is _FINALIZED
        had_prior_failure = self._had_failure
        if not was_finalized:
            self._state = _ABORTED
        cleanup_error = self._release_resources()
        self._state = _CLOSED
        if cleanup_error is not None and not had_prior_failure:
            if was_finalized:
                raise _build_load_error("FINALIZE_INVALID_STATE") from cleanup_error
            raise _build_load_error("SESSION_ABORTED") from cleanup_error

    def __enter__(self) -> FrvSourceLoadSession:
        return self

    def __exit__(self, exc_type, exc, traceback) -> Literal[False]:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:  # noqa: BLE001 - must not mask the active exception
                pass
        return False

    # ------------------------------------------------------------------
    # Reading and validation
    # ------------------------------------------------------------------

    def _read_and_validate_header(self, las_header: Any):
        if las_header.are_points_compressed:
            raise _build_load_error("FORMAT_REJECTED")
        if getattr(las_header, "evlrs", None):
            raise _build_load_error("FORMAT_REJECTED")
        non_crs_vlr_count = 0
        for vlr in getattr(las_header, "vlrs", ()):
            if str(getattr(vlr, "user_id", "")) == "LASF_Projection":
                raise _build_load_error("COORDINATE_AUTHORITY_REJECTED")
            if type(vlr).__name__ in _KNOWN_CRS_VLR_TYPES:
                raise _build_load_error("COORDINATE_AUTHORITY_REJECTED")
            non_crs_vlr_count += 1
        try:
            parsed_crs = las_header.parse_crs()
        except Exception as exc:
            raise _build_load_error("COORDINATE_AUTHORITY_REJECTED") from exc
        if parsed_crs is not None:
            raise _build_load_error("COORDINATE_AUTHORITY_REJECTED")
        if int(las_header.point_count) <= 0:
            raise _build_load_error("ZERO_POINT_SOURCE")
        try:
            header_metadata = self._build_header_metadata(las_header, non_crs_vlr_count)
        except Exception as exc:
            raise _build_load_error("HEADER_INVALID") from exc
        return header_metadata, las_header

    def _build_header_metadata(
        self, las_header: Any, non_crs_vlr_count: int
    ) -> FrvSourceHeaderMetadata:
        point_format = las_header.point_format
        # laspy 2.7 ``dimension_names`` reports standard AND extra dimensions
        # together; use the authoritative standard-only API so Extra Bytes
        # dimensions (e.g. NormalX/NormalY/NormalZ) are classified exclusively
        # as ``extra_dimensions`` and never leak into ``dimensions``.
        standard_dimension_names = tuple(
            str(name) for name in point_format.standard_dimension_names
        )
        standard_names = {
            name.lower()
            for name in standard_dimension_names
            if name.lower() not in ("x", "y", "z")
        }
        dimensions = tuple(sorted(standard_names))
        extra_dimensions = tuple(
            sorted(str(name).lower() for name in point_format.extra_dimension_names)
        )
        return FrvSourceHeaderMetadata(
            schema_version=FRV_SOURCE_PROVIDER_HEADER_SCHEMA_VERSION,
            source_format=FrvSourceFormat.LAS,
            las_version=str(las_header.version),
            point_format_id=int(point_format.id),
            point_record_length_bytes=int(point_format.size),
            point_count=int(las_header.point_count),
            scale_xyz=(
                float(las_header.scales[0]),
                float(las_header.scales[1]),
                float(las_header.scales[2]),
            ),
            offset_xyz=(
                float(las_header.offsets[0]),
                float(las_header.offsets[1]),
                float(las_header.offsets[2]),
            ),
            min_xyz=(
                float(las_header.mins[0]),
                float(las_header.mins[1]),
                float(las_header.mins[2]),
            ),
            max_xyz=(
                float(las_header.maxs[0]),
                float(las_header.maxs[1]),
                float(las_header.maxs[2]),
            ),
            dimensions=dimensions,
            extra_dimensions=extra_dimensions,
            compression_state=FrvSourceCompressionState.UNCOMPRESSED,
            crs_authority_state=FrvSourceCrsAuthorityState.ABSENT,
            non_crs_vlr_count=non_crs_vlr_count,
            non_crs_evlr_count=0,
            vlr_policy_state=FrvSourceVlrPolicyState.NO_CRS_AUTHORITY_DETECTED,
        )

    def _build_chunk(
        self,
        raw: Any,
        point_count: int,
        start_index: int,
        chunk_id: int,
        *,
        is_last: bool,
    ) -> FrvSourceChunk:
        try:
            xyz = np.stack(
                [np.asarray(raw.x), np.asarray(raw.y), np.asarray(raw.z)], axis=1
            )
            if not np.isfinite(xyz).all():
                raise _build_load_error("NONFINITE_COORDINATE")
            xyz.setflags(write=False)
            original_index = np.arange(start_index, start_index + point_count, dtype="<i8")
            original_index.setflags(write=False)
            attributes: dict[str, np.ndarray] = {}
            for dimension in self._provider_config.pass_through_dimensions:
                name = dimension.value
                if not hasattr(raw, name):
                    raise _build_load_error("FORMAT_REJECTED")
                values = np.asarray(getattr(raw, name))
                dtype = _FRV_SOURCE_ATTRIBUTE_DTYPES[dimension]
                array = np.ascontiguousarray(values.astype(dtype, copy=True))
                array.setflags(write=False)
                attributes[name] = array
            return FrvSourceChunk(
                source_id=self._reference.source_identity.source_id,
                chunk_id=chunk_id,
                start_index=start_index,
                point_count=point_count,
                xyz=xyz,
                original_index=original_index,
                attributes=MappingProxyType(attributes),
                is_last=is_last,
            )
        except FrvSourceLoadError:
            raise
        except Exception as exc:
            raise _build_load_error("SESSION_ABORTED") from exc

    def _accumulate_bounds(self, chunk, bounds_min, bounds_max):
        chunk_min = chunk.xyz.min(axis=0)
        chunk_max = chunk.xyz.max(axis=0)
        if bounds_min is None:
            return chunk_min, chunk_max
        return np.minimum(bounds_min, chunk_min), np.maximum(bounds_max, chunk_max)

    def _feed_frvpcd1(self, chunk: FrvSourceChunk) -> None:
        if self._frvpcd1 is None:
            return
        for row in range(chunk.point_count):
            original_index = int(chunk.original_index[row])
            x = float(chunk.xyz[row, 0]) + 0.0
            y = float(chunk.xyz[row, 1]) + 0.0
            z = float(chunk.xyz[row, 2]) + 0.0
            self._frvpcd1.update(struct.pack("<qddd", original_index, x, y, z))

    @staticmethod
    def _freeze_bounds(bounds_min, bounds_max):
        if bounds_min is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        return (
            (float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2])),
            (float(bounds_max[0]), float(bounds_max[1]), float(bounds_max[2])),
        )

    @staticmethod
    def _hash_handle(handle) -> tuple[str, int]:
        hasher = hashlib.sha256()
        size = 0
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            hasher.update(block)
            size += len(block)
        return hasher.hexdigest(), size

    def _check_toctou(self, final_sha: str, final_size: int) -> None:
        if final_sha != self._initial_sha or final_size != self._initial_size:
            raise _build_load_error("TOCTOU_CHANGED")
        component = self._reference.artifact_components[0]
        if final_sha != component.sha256 or final_size != component.size_bytes:
            raise _build_load_error("TOCTOU_CHANGED")
        identity = self._reference.source_identity
        if final_sha != identity.source_file_sha256:
            raise _build_load_error("TOCTOU_CHANGED")

    def _build_proof(self) -> FrvSourceProviderProof:
        assert self._total_points is not None
        assert self._streamed_bounds is not None
        identity = self._reference.source_identity
        canonical_digest = None
        canonical_spec_digest = None
        if self._frvpcd1 is not None:
            canonical_digest = self._frvpcd1.hexdigest()
            canonical_spec_digest = self._canonical_spec.spec_digest
        return FrvSourceProviderProof(
            schema_version=FRV_SOURCE_PROVIDER_SCHEMA_VERSION,
            provider_key=FRV_SOURCE_PROVIDER_KEY,
            provider_role=FrvA2ProviderRole.SOURCE_SCAN,
            interface_version=FRV_SOURCE_PROVIDER_INTERFACE_VERSION,
            algorithm_version=FRV_SOURCE_PROVIDER_ALGORITHM_VERSION,
            configuration_digest=self._frv_configuration.configuration_digest,
            provider_config_digest=self._provider_config.provider_config_digest,
            artifact_reference_id=self._reference.artifact_reference_id,
            artifact_components=self._reference.artifact_components,
            source_content_identity_digest=identity.source_content_identity_digest,
            source_identity_record_digest=identity.source_identity_record_digest,
            source_point_count=self._total_points,
            source_bounds_min_xyz=self._streamed_bounds[0],
            source_bounds_max_xyz=self._streamed_bounds[1],
            coordinate_frame="LOCAL_CARTESIAN_XYZ",
            units="m",
            origin_xyz=tuple(identity.origin_context),
            transform_declaration="identity",
            domain_kind=FrvA2DomainKind.FULL_SOURCE,
            domain_count=self._total_points,
            original_index_domain_digest=_stream_index_domain_digest(
                self._total_points, identity.source_content_identity_digest
            ),
            sample_index_domain_digest=None,
            canonical_point_content_digest=canonical_digest,
            canonical_point_content_spec_digest=canonical_spec_digest,
            header=self._header_metadata,
            pass_through_dimensions=self._provider_config.pass_through_dimensions,
        )

    def _mark_aborted(self) -> None:
        if self._state not in (_FINALIZED, _CLOSED, _ABORTED):
            self._state = _ABORTED

    def _abort(self) -> None:
        self._mark_aborted()
        self._release_resources()
        self._had_failure = True

    def _close_reader(self) -> BaseException | None:
        if self._reader is None:
            return None
        try:
            self._reader.close()
        except Exception as exc:  # noqa: BLE001 - cleanup path
            self._reader = None
            return exc
        self._reader = None
        return None

    def _close_handle(self) -> BaseException | None:
        if self._handle is None:
            return None
        try:
            self._handle.close()
        except Exception as exc:  # noqa: BLE001 - cleanup path
            self._handle = None
            return exc
        self._handle = None
        return None

    def _release_resources(self) -> BaseException | None:
        first_error = self._close_reader()
        handle_error = self._close_handle()
        if handle_error is not None and first_error is None:
            first_error = handle_error
        return first_error


def load_frv_source(
    reference: FrvSourceProviderReference,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvSourceProviderConfig,
    chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvSourceLoadSession:
    """Open a single-use P-SOURCE load session over *reference*.

    The reference and configuration bindings are fully validated before any
    file is opened; validation failure is normalized to ``REFERENCE_INVALID``.
    The returned object is the private ``_FrvSourceLoadSession`` (structural
    ``FrvSourceLoadSession``), and only ``load_frv_source`` is exported from
    ``algorithms.io``.
    """

    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(frv_configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")
    if type(provider_config) is not FrvSourceProviderConfig:
        raise TypeError("provider_config must be a FrvSourceProviderConfig")
    if type(chunk_size) is not int:
        raise TypeError("chunk_size must be a built-in int")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    try:
        validate_frv_source_provider_reference(
            reference,
            provider_config=provider_config,
            expected_configuration_digest=frv_configuration.configuration_digest,
        )
    except Exception as exc:
        raise _build_load_error("REFERENCE_INVALID") from exc
    return _FrvSourceLoadSession(reference, frv_configuration, provider_config, chunk_size)
