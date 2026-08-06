"""FRV-A2 P-WIRE runtime provider (PLAN-FROZEN).

P-WIRE consumes a finalized P-SOURCE result and an approved, source-bound
``FrvWireAuthorityReference`` and runs exactly one independent
``load_frv_source`` session.  The geometry index is built from the verified
authority polyline edges before the source is opened; every source row is
decided exactly once and produces one ``FrvWireChunk`` per source chunk over
the complete FULL_SOURCE domain.  No direct LAS access, no second source pass,
no inference of full-resolution relation from A4 discrete assignment.
"""

from __future__ import annotations

import hashlib
import math
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
    FrvWireProviderState,
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
from core.contracts.frv_wire import (
    _FRV_WIRE_FINALIZED_CAPABILITY,
    _FRV_WIRE_SOURCE_PROOF_DIGEST_DOMAIN,
    FRV_WIRE_ALGORITHM_VERSION,
    FRV_WIRE_PROOF_SCHEMA_VERSION,
    FRV_WIRE_PROVIDER_INTERFACE_VERSION,
    FRV_WIRE_PROVIDER_KEY,
    FrvWireAuthorityManifest,
    FrvWireAuthorityMode,
    FrvWireAuthorityReference,
    FrvWireChunk,
    FrvWireLoadError,
    FrvWireLoadSession,
    FrvWirePointRelationState,
    FrvWireProviderConfig,
    FrvWireProviderProof,
    FrvWireProviderResult,
    _component_sort_key,
    _create_finalized_frv_wire_provider_result,
    frv_wire_geometry_content_digest,
    frv_wire_point_content_digest,
    validate_frv_wire_authority_manifest,
    validate_frv_wire_provider_config,
)

from algorithms.io.source_provider import load_frv_source
from algorithms.wires.reconstruct import load_wire_models

# Private session-state markers (never exposed).
_CREATED = "CREATED"
_ITERATING = "ITERATING"
_STREAM_COMPLETE = "STREAM_COMPLETE"
_FINALIZED = "FINALIZED"
_ABORTED = "ABORTED"
_CLOSED = "CLOSED"

# Frozen plan section 4.5 mapping keys -> failure-record tuple.
_WIRE_FAILURE_MAP: dict[
    str, tuple[FrvA2FailurePhase, str, FrvErrorCode, FrvReasonFlag, FrvQualityFlag]
] = {
    "WIRE_AUTHORITY_INVALID": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "wire_authority",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "WIRE_INVENTORY_INVALID": (
        FrvA2FailurePhase.PRODUCTION_POLICY,
        "wire_inventory",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
    ),
    "WIRE_IDENTITY_MISMATCH": (
        FrvA2FailurePhase.IDENTITY_CONTEXT,
        "wire_authority",
        FrvErrorCode.FRV_WIRE_IDENTITY_MISMATCH,
        FrvReasonFlag.WIRE_CONFLICT,
        FrvQualityFlag.WIRE_IDENTITY_MISMATCH,
    ),
    "WIRE_ARTIFACT_FAILURE": (
        FrvA2FailurePhase.STRUCTURE_INTEGRITY,
        "wire_artifact",
        FrvErrorCode.FRV_WIRE_GEOMETRY_UNAVAILABLE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.WIRE_GEOMETRY_UNAVAILABLE,
    ),
    "WIRE_GEOMETRY_FAILURE": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "wire_geometry",
        FrvErrorCode.FRV_WIRE_GEOMETRY_UNAVAILABLE,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.WIRE_GEOMETRY_UNAVAILABLE,
    ),
    "WIRE_INDEX_FAILURE": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "wire_index",
        FrvErrorCode.FRV_PROVIDER_INVALID,
        FrvReasonFlag.PROVIDER_UNCERTAINTY,
        FrvQualityFlag.PROVIDER_INVALID,
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
    "INDEX_OUT_OF_RANGE": (
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
    "NONFINITE_DISTANCE": (
        FrvA2FailurePhase.SEMANTIC_READINESS,
        "centerline_distance",
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


def _build_wire_error(key: str) -> FrvWireLoadError:
    phase, field_name, error_code, reason_flags, quality_flag = _WIRE_FAILURE_MAP[key]
    detail = (
        f"{phase.value}:{FrvA2ProviderRole.WIRE.value}:{FRV_WIRE_PROVIDER_KEY}:"
        f"{field_name}:{error_code.value}:{int(reason_flags)}"
    )
    record = FrvA2FailureRecord(
        phase,
        FrvA2ProviderRole.WIRE,
        FRV_WIRE_PROVIDER_KEY,
        field_name,
        error_code,
        int(reason_flags),
        detail,
    )
    return FrvWireLoadError(failure_record=record, quality_flags=int(quality_flag))


def _normalize_zero(value: float) -> float:
    return 0.0 if value == 0.0 else float(value)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


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
        raise _build_wire_error("WIRE_AUTHORITY_INVALID")
    return FrvSourceProviderConfig(
        schema_version=FRV_SOURCE_PROVIDER_CONFIG_SCHEMA_VERSION,
        source_model=FrvSourceModel.SINGLE_FILE_ONLY,
        coordinate_conversion_policy=FrvSourceCoordinateConversionPolicy.FORBID,
        invalid_point_policy=FrvSourceInvalidPointPolicy.FAIL_CLOSED,
        canonical_content_policy=FrvSourceCanonicalContentPolicy.VERIFY_IF_DECLARED,
        pass_through_dimensions=pass_through_dimensions,
        provider_config_digest=provider_config_digest,
    )


class _Edge:
    __slots__ = (
        "wire_id",
        "segment_index",
        "edge_index",
        "ax",
        "ay",
        "az",
        "bx",
        "by",
        "bz",
    )

    def __init__(
        self,
        wire_id: int,
        segment_index: int,
        edge_index: int,
        ax: float,
        ay: float,
        az: float,
        bx: float,
        by: float,
        bz: float,
    ) -> None:
        self.wire_id = wire_id
        self.segment_index = segment_index
        self.edge_index = edge_index
        self.ax = ax
        self.ay = ay
        self.az = az
        self.bx = bx
        self.by = by
        self.bz = bz


class _EdgeIndex:
    """Half-open 1.50 m spatial index over expanded polyline-edge AABBs."""

    def __init__(
        self,
        min_xyz: tuple[float, float, float],
        cell_size: float,
        halo: float,
        max_edges_per_cell: int,
    ) -> None:
        self._min_x, self._min_y, self._min_z = min_xyz
        self._cell_size = cell_size
        self._halo = halo
        self._max_edges_per_cell = max_edges_per_cell
        self._cells: dict[tuple[int, int, int], list[_Edge]] = {}
        self._edge_count = 0

    def cell_key(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            math.floor((x - self._min_x) / self._cell_size),
            math.floor((y - self._min_y) / self._cell_size),
            math.floor((z - self._min_z) / self._cell_size),
        )

    def insert(self, edge: _Edge) -> None:
        halo = self._halo
        lo_x = math.floor((min(edge.ax, edge.bx) - halo - self._min_x) / self._cell_size)
        hi_x = math.floor((max(edge.ax, edge.bx) + halo - self._min_x) / self._cell_size)
        lo_y = math.floor((min(edge.ay, edge.by) - halo - self._min_y) / self._cell_size)
        hi_y = math.floor((max(edge.ay, edge.by) + halo - self._min_y) / self._cell_size)
        lo_z = math.floor((min(edge.az, edge.bz) - halo - self._min_z) / self._cell_size)
        hi_z = math.floor((max(edge.az, edge.bz) + halo - self._min_z) / self._cell_size)
        for ix in range(lo_x, hi_x + 1):
            for iy in range(lo_y, hi_y + 1):
                for iz in range(lo_z, hi_z + 1):
                    key = (ix, iy, iz)
                    cell = self._cells.get(key)
                    if cell is None:
                        cell = []
                        self._cells[key] = cell
                    cell.append(edge)
                    if len(cell) > self._max_edges_per_cell:
                        raise _build_wire_error("WIRE_INDEX_FAILURE")
        self._edge_count += 1

    def edge_count(self) -> int:
        return self._edge_count

    def halo_cell_keys(self, x: float, y: float, z: float) -> list[tuple[int, int, int]]:
        halo = self._halo
        ix_lo = math.floor((x - halo - self._min_x) / self._cell_size)
        ix_hi = math.floor((x + halo - self._min_x) / self._cell_size)
        iy_lo = math.floor((y - halo - self._min_y) / self._cell_size)
        iy_hi = math.floor((y + halo - self._min_y) / self._cell_size)
        iz_lo = math.floor((z - halo - self._min_z) / self._cell_size)
        iz_hi = math.floor((z + halo - self._min_z) / self._cell_size)
        keys = []
        for ix in range(ix_lo, ix_hi + 1):
            for iy in range(iy_lo, iy_hi + 1):
                for iz in range(iz_lo, iz_hi + 1):
                    keys.append((ix, iy, iz))
        return keys

    def query(
        self, x: float, y: float, z: float
    ) -> tuple[float, int, int, int] | None:
        """Return ``(d2, wire_id, segment_index, edge_index)`` of the nearest edge."""

        best: tuple[float, int, int, int] | None = None
        for key in self.halo_cell_keys(x, y, z):
            for edge in self._cells.get(key, ()):
                d2 = _segment_distance_squared(x, y, z, edge)
                if not math.isfinite(d2):
                    raise _build_wire_error("NONFINITE_DISTANCE")
                candidate = (d2, edge.wire_id, edge.segment_index, edge.edge_index)
                if best is None or candidate < best:
                    best = candidate
        return best


def _segment_distance_squared(
    x: float, y: float, z: float, edge: _Edge
) -> float:
    ux = edge.bx - edge.ax
    uy = edge.by - edge.ay
    uz = edge.bz - edge.az
    pax = x - edge.ax
    pay = y - edge.ay
    paz = z - edge.az
    denom = ux * ux + uy * uy + uz * uz
    if denom == 0.0:
        # Zero-length edges are rejected during construction; the defensive
        # branch keeps a degenerate edge from producing NaN in a query.
        t = 0.0
    else:
        t = (pax * ux + pay * uy + paz * uz) / denom
        t = max(0.0, min(1.0, t))
    qx = edge.ax + t * ux
    qy = edge.ay + t * uy
    qz = edge.az + t * uz
    dx = x - qx
    dy = y - qy
    dz = z - qz
    return dx * dx + dy * dy + dz * dz


class _FrvWireLoadSession:
    """Single-use one-source-pass P-WIRE session (private; not exported)."""

    def __init__(
        self,
        reference: FrvSourceProviderReference,
        source_result: FrvSourceProviderResult,
        authority_reference: FrvWireAuthorityReference,
        frv_configuration: FrvConfiguration,
        provider_config: FrvWireProviderConfig,
        source_chunk_size: int,
    ) -> None:
        self._reference = reference
        self._source_result = source_result
        self._authority_reference = authority_reference
        self._frv_configuration = frv_configuration
        self._provider_config = provider_config
        self._source_chunk_size = source_chunk_size
        self._state = _CREATED
        self._had_failure = False
        self._result: FrvWireProviderResult | None = None
        self._source_point_count = int(source_result.proof.source_point_count)
        self._expected_source_id = str(source_result.source_identity.source_id)
        self._active_source_sessions: list[Any] = []
        self._pending_chunks: list[tuple] = []
        self._wire_point_content_digest: str | None = None
        self._edge_index: _EdgeIndex | None = None
        self._geometry_digest: str = ""
        self._w1_result: Any = None
        self._w1_arrays: dict[str, np.ndarray] | None = None
        self._approved_wire_count = 0
        self._relation_valid_count = 0
        self._not_applicable_count = 0
        self._outside_count = 0
        self._body_count = 0
        self._proximity_count = 0
        self._conflict_count = 0
        self._has_in_domain = False
        self._min_distance = 0.0
        self._max_distance = 0.0
        self._pre_pass_validate()

    # ------------------------------------------------------------------
    # Pre-pass validation (frozen first-failure order)
    # ------------------------------------------------------------------

    def _pre_pass_validate(self) -> None:
        if type(self._source_result) is not FrvSourceProviderResult:
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        if (
            getattr(self._source_result, "_finalized_capability", None)
            is not _FRV_SOURCE_FINALIZED_CAPABILITY
        ):
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        try:
            validate_finalized_frv_source_provider_result(self._source_result)
        except Exception as exc:  # noqa: BLE001 - validator failure boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        if type(self._reference) is not FrvSourceProviderReference:
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        try:
            validate_configuration(self._frv_configuration)
        except Exception as exc:  # noqa: BLE001 - config validation boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        if (
            self._frv_configuration.wire_provider_interface_version is None
            or self._frv_configuration.wire_provider_config_digest is None
        ):
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        try:
            validate_frv_wire_provider_config(self._provider_config)
        except Exception as exc:  # noqa: BLE001 - config validation boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        if (
            self._provider_config.interface_version
            != self._frv_configuration.wire_provider_interface_version
            or self._provider_config.provider_config_digest
            != self._frv_configuration.wire_provider_config_digest
        ):
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        authority_reference = self._authority_reference
        if type(authority_reference) is not FrvWireAuthorityReference:
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        try:
            validate_frv_wire_authority_manifest(authority_reference.manifest)
        except Exception as exc:  # noqa: BLE001 - manifest validation boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        manifest = authority_reference.manifest
        if manifest.inventory_complete is not True:
            raise _build_wire_error("WIRE_INVENTORY_INVALID")
        try:
            self._reconstructed_config = _reconstruct_frv_source_provider_config(
                self._source_result.proof
            )
        except FrvWireLoadError:
            raise
        except Exception as exc:  # noqa: BLE001 - reconstruction boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        try:
            validate_frv_source_provider_reference(
                self._reference,
                provider_config=self._reconstructed_config,
                expected_configuration_digest=self._frv_configuration.configuration_digest,
            )
        except Exception as exc:  # noqa: BLE001 - reference binding boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        source_proof = self._source_result.proof
        if source_proof.provider_role is not FrvA2ProviderRole.SOURCE_SCAN:
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if source_proof.provider_key != FRV_SOURCE_PROVIDER_KEY:
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if source_proof.domain_kind is not FrvA2DomainKind.FULL_SOURCE:
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if source_proof.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if source_proof.units != "m":
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if source_proof.transform_declaration != "identity":
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if source_proof.source_point_count <= 0:
            raise _build_wire_error("POINT_COUNT_MISMATCH")
        self._verify_authority_identity(manifest, source_proof)
        self._verify_authority_component()
        self._approved_wire_count = len(manifest.approved_wire_ids)
        if manifest.authority_mode is FrvWireAuthorityMode.W1_MODEL_SET:
            self._load_and_verify_w1(manifest)
        else:
            self._geometry_digest = frv_wire_geometry_content_digest(
                source_content_identity_digest=manifest.source_content_identity_digest,
                source_identity_record_digest=manifest.source_identity_record_digest,
                origin_xyz=manifest.origin_xyz,
                edges=[],
            )

    def _verify_authority_identity(
        self, manifest: FrvWireAuthorityManifest, source_proof: FrvSourceProviderProof
    ) -> None:
        if manifest.source_content_identity_digest != source_proof.source_content_identity_digest:
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if manifest.source_identity_record_digest != source_proof.source_identity_record_digest:
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if manifest.source_id != self._source_result.source_identity.source_id:
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if tuple(manifest.origin_xyz) != tuple(source_proof.origin_xyz):
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if manifest.coordinate_frame != "LOCAL_CARTESIAN_XYZ":
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if manifest.units != "m":
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
        if manifest.transform_declaration != "identity":
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH")

    def _verify_authority_component(self) -> None:
        component = self._authority_reference.authority_manifest_component
        path = self._authority_reference.authority_manifest_path
        try:
            size = path.stat().st_size
            digest = _sha256_file(path)
        except OSError as exc:  # pragma: no cover - IO boundary
            raise _build_wire_error("WIRE_AUTHORITY_INVALID") from exc
        if size != component.size_bytes or digest != component.sha256:
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")

    def _verify_component_file(
        self, path: Path, component: Any
    ) -> None:
        try:
            size = path.stat().st_size
            digest = _sha256_file(path)
        except OSError as exc:  # pragma: no cover - IO boundary
            raise _build_wire_error("WIRE_ARTIFACT_FAILURE") from exc
        if size != component.size_bytes or digest != component.sha256:
            raise _build_wire_error("WIRE_ARTIFACT_FAILURE")

    def _load_and_verify_w1(self, manifest: FrvWireAuthorityManifest) -> None:
        ref = self._authority_reference
        assert ref.w1_json_path is not None and ref.w1_npz_path is not None
        components = {
            component.component_key: component
            for component in manifest.w1_artifact_components
        }
        try:
            self._w1_result, self._w1_arrays = load_wire_models(
                ref.w1_npz_path, ref.w1_json_path
            )
        except Exception as exc:  # noqa: BLE001 - W1 loader boundary
            raise _build_wire_error("WIRE_ARTIFACT_FAILURE") from exc
        self._verify_component_file(ref.w1_json_path, components["manifest_json"])
        self._verify_component_file(ref.w1_npz_path, components["payload_npz"])
        result = self._w1_result
        if result.source_id != manifest.source_id:
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        if result.manifest_source_id != manifest.source_id:
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        arrays = self._w1_arrays
        assert arrays is not None
        wire_ids = [int(value) for value in arrays["wire_id"]]
        if wire_ids != list(manifest.approved_wire_ids):
            raise _build_wire_error("WIRE_AUTHORITY_INVALID")
        edges = self._build_edges(arrays, manifest)
        self._geometry_digest = frv_wire_geometry_content_digest(
            source_content_identity_digest=manifest.source_content_identity_digest,
            source_identity_record_digest=manifest.source_identity_record_digest,
            origin_xyz=manifest.origin_xyz,
            edges=edges,
        )
        if self._geometry_digest != manifest.geometry_content_digest:
            raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
        index = _EdgeIndex(
            tuple(float(value) for value in self._source_result.proof.source_bounds_min_xyz),
            self._provider_config.spatial_index_cell_size_m,
            self._provider_config.proximity_radius_m,
            self._provider_config.max_edges_per_index_cell,
        )
        for edge in self._edge_records:
            index.insert(edge)
        self._edge_index = index

    def _build_edges(self, arrays: dict[str, np.ndarray], manifest: FrvWireAuthorityManifest):
        wire_status = arrays["wire_status"]
        continuous = arrays["wire_continuous_distance_ready"]
        wire_offsets = arrays["wire_segment_offsets"]
        segment_wire_id = arrays["segment_wire_id"]
        segment_index = arrays["segment_index"]
        segment_ready = arrays["segment_distance_ready"]
        curve_offsets = arrays["sampled_curve_offsets"]
        curve_xyz = arrays["sampled_curve_xyz"]
        origin = tuple(manifest.origin_xyz)
        edges: list[tuple[int, int, int, float, float, float, float, float, float]] = []
        records: list[_Edge] = []
        for position in range(len(arrays["wire_id"])):
            if int(wire_status[position]) != 0:
                raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
            if not bool(continuous[position]):
                raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
            seg_start = int(wire_offsets[position])
            seg_stop = int(wire_offsets[position + 1])
            for segment_position in range(seg_start, seg_stop):
                if int(segment_wire_id[segment_position]) != int(arrays["wire_id"][position]):
                    raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
                if not bool(segment_ready[segment_position]):
                    raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
                curve_start = int(curve_offsets[segment_position])
                curve_stop = int(curve_offsets[segment_position + 1])
                if curve_stop - curve_start < 2:
                    raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
                seg = int(segment_index[segment_position])
                for edge_position in range(curve_start, curve_stop - 1):
                    a = curve_xyz[edge_position]
                    b = curve_xyz[edge_position + 1]
                    ax = float(a[0]) + origin[0]
                    ay = float(a[1]) + origin[1]
                    az = float(a[2]) + origin[2]
                    bx = float(b[0]) + origin[0]
                    by = float(b[1]) + origin[1]
                    bz = float(b[2]) + origin[2]
                    values = (ax, ay, az, bx, by, bz)
                    if not all(math.isfinite(value) for value in values):
                        raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
                    if (ax == bx and ay == by and az == bz):
                        raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
                    edge = _Edge(
                        int(arrays["wire_id"][position]),
                        seg,
                        edge_position - curve_start,
                        ax,
                        ay,
                        az,
                        bx,
                        by,
                        bz,
                    )
                    records.append(edge)
                    edges.append(
                        (
                            int(arrays["wire_id"][position]),
                            seg,
                            edge_position - curve_start,
                            ax,
                            ay,
                            az,
                            bx,
                            by,
                            bz,
                        )
                    )
        if not edges:
            raise _build_wire_error("WIRE_GEOMETRY_FAILURE")
        self._edge_records = records
        return edges

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def iter_chunks(self) -> Iterator[FrvWireChunk]:
        if self._state is not _CREATED:
            raise _build_wire_error("REPEATED_ITERATOR")
        self._state = _ITERATING
        return self._stream_wire_chunks()

    def _stream_wire_chunks(self) -> Iterator[FrvWireChunk]:
        try:
            self._run_source_pass()
            assembled = [self._assemble_wire_chunk(data) for data in self._pending_chunks]
            self._check_wire_stream_continuity(assembled)
            self._wire_point_content_digest = frv_wire_point_content_digest(
                source_content_identity_digest=self._source_result.proof.source_content_identity_digest,
                source_identity_record_digest=self._source_result.proof.source_identity_record_digest,
                authority_manifest_digest=self._authority_reference.manifest.manifest_digest,
                wire_geometry_content_digest=self._geometry_digest,
                provider_config_digest=self._provider_config.provider_config_digest,
                source_point_count=self._source_point_count,
                chunks=assembled,
            )
            yield from assembled
            self._state = _STREAM_COMPLETE
        except GeneratorExit:
            self._abort()
            raise
        except FrvWireLoadError:
            self._abort()
            raise
        except Exception as exc:
            self._abort()
            raise _build_wire_error("SOURCE_PASS_INCOMPLETE") from exc

    def finalize(self) -> FrvWireProviderResult:
        if self._state in (_CREATED, _ITERATING, _ABORTED):
            raise _build_wire_error("PREMATURE_FINALIZE")
        if self._state is not _STREAM_COMPLETE:
            raise _build_wire_error("REPEATED_FINALIZE")
        try:
            self._verify_authority_component()
            proof = self._build_proof()
            result = _create_finalized_frv_wire_provider_result(
                proof,
                self._source_result.source_identity,
                capability=_FRV_WIRE_FINALIZED_CAPABILITY,
            )
            self._result = result
            self._state = _FINALIZED
            return result
        except FrvWireLoadError:
            self._abort()
            raise
        except ValueError as exc:
            self._abort()
            raise _build_wire_error("WIRE_IDENTITY_MISMATCH") from exc
        except Exception as exc:
            self._abort()
            raise _build_wire_error("DIGEST_MISMATCH") from exc

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
            raise _build_wire_error("CLEANUP_FAILURE") from cleanup_error

    def __enter__(self) -> FrvWireLoadSession:
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
    # Source pass
    # ------------------------------------------------------------------

    def _run_source_pass(self) -> None:
        self._pending_chunks = []
        session = load_frv_source(
            self._reference,
            frv_configuration=self._frv_configuration,
            provider_config=self._reconstructed_config,
            chunk_size=self._source_chunk_size,
        )
        self._active_source_sessions.append(session)
        try:
            with session:
                for source_chunk in session.iter_chunks():
                    if source_chunk.source_id != self._expected_source_id:
                        raise _build_wire_error("WIRE_IDENTITY_MISMATCH")
                    data = self._build_wire_chunk_data(source_chunk)
                    self._pending_chunks.append(data)
                result = session.finalize()
            self._check_source_result_matches(result)
        finally:
            if session in self._active_source_sessions:
                self._active_source_sessions.remove(session)

    def _build_wire_chunk_data(self, source_chunk: Any) -> tuple:
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
            "provider_state": np.empty(point_count, dtype="|u1"),
            "relation_state": np.empty(point_count, dtype="|u1"),
            "nearest_wire_id": np.empty(point_count, dtype="<i4"),
            "nearest_segment_index": np.empty(point_count, dtype="<i4"),
            "nearest_edge_index": np.empty(point_count, dtype="<i4"),
            "centerline_distance_m": np.empty(point_count, dtype="<f8"),
            "reason_flags": np.zeros(point_count, dtype="<u4"),
            "quality_flags": np.zeros(point_count, dtype="<u4"),
        }
        config = self._provider_config
        r0 = float(config.wire_body_possible_radius_m)
        r1 = float(config.vegetation_conflict_radius_m)
        r2 = float(config.proximity_radius_m)
        empty = (
            self._authority_reference.manifest.authority_mode
            is FrvWireAuthorityMode.EMPTY_INVENTORY
        )
        for row in range(point_count):
            arrays["original_index"][row] = source_chunk.original_index[row]
            if empty:
                arrays["provider_state"][row] = int(
                    FrvWireProviderState.WIRE_NOT_APPLICABLE_PROVEN
                )
                arrays["relation_state"][row] = int(
                    FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN
                )
                arrays["nearest_wire_id"][row] = -1
                arrays["nearest_segment_index"][row] = -1
                arrays["nearest_edge_index"][row] = -1
                arrays["centerline_distance_m"][row] = 0.0
                self._not_applicable_count += 1
                self._outside_count += 1
                continue
            arrays["provider_state"][row] = int(FrvWireProviderState.WIRE_RELATION_VALID)
            self._relation_valid_count += 1
            x = float(source_chunk.xyz[row, 0])
            y = float(source_chunk.xyz[row, 1])
            z = float(source_chunk.xyz[row, 2])
            best = None
            if self._edge_index is not None:
                best = self._edge_index.query(x, y, z)
            if best is None:
                arrays["relation_state"][row] = int(
                    FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN
                )
                arrays["nearest_wire_id"][row] = -1
                arrays["nearest_segment_index"][row] = -1
                arrays["nearest_edge_index"][row] = -1
                arrays["centerline_distance_m"][row] = 0.0
                self._outside_count += 1
                continue
            d2, wire_id, segment_index, edge_index = best
            distance = math.sqrt(d2)
            if distance > r2:
                arrays["relation_state"][row] = int(
                    FrvWirePointRelationState.OUTSIDE_INFLUENCE_DOMAIN
                )
                arrays["nearest_wire_id"][row] = -1
                arrays["nearest_segment_index"][row] = -1
                arrays["nearest_edge_index"][row] = -1
                arrays["centerline_distance_m"][row] = 0.0
                self._outside_count += 1
                continue
            if distance <= r0:
                relation = int(FrvWirePointRelationState.WIRE_BODY_POSSIBLE)
                self._body_count += 1
            elif distance <= r1:
                relation = int(FrvWirePointRelationState.WIRE_VEGETATION_CONFLICT)
                self._conflict_count += 1
            else:
                relation = int(FrvWirePointRelationState.WIRE_PROXIMITY_ONLY)
                self._proximity_count += 1
            arrays["relation_state"][row] = relation
            arrays["nearest_wire_id"][row] = wire_id
            arrays["nearest_segment_index"][row] = segment_index
            arrays["nearest_edge_index"][row] = edge_index
            arrays["centerline_distance_m"][row] = _normalize_zero(distance)
            if not self._has_in_domain:
                self._has_in_domain = True
                self._min_distance = distance
                self._max_distance = distance
            else:
                if distance < self._min_distance:
                    self._min_distance = distance
                if distance > self._max_distance:
                    self._max_distance = distance
        return (scalars, arrays)

    def _assemble_wire_chunk(self, data: tuple) -> FrvWireChunk:
        scalars, arrays = data
        frozen = {}
        for name, array in arrays.items():
            frozen[name] = np.ascontiguousarray(array)
            frozen[name].setflags(write=False)
        return FrvWireChunk(
            source_id=scalars["source_id"],
            chunk_id=scalars["chunk_id"],
            start_index=scalars["start_index"],
            point_count=scalars["point_count"],
            original_index=frozen["original_index"],
            provider_state=frozen["provider_state"],
            relation_state=frozen["relation_state"],
            nearest_wire_id=frozen["nearest_wire_id"],
            nearest_segment_index=frozen["nearest_segment_index"],
            nearest_edge_index=frozen["nearest_edge_index"],
            centerline_distance_m=frozen["centerline_distance_m"],
            reason_flags=frozen["reason_flags"],
            quality_flags=frozen["quality_flags"],
            is_last=scalars["is_last"],
        )

    # ------------------------------------------------------------------
    # Continuity, source-result match, and proof
    # ------------------------------------------------------------------

    def _check_wire_stream_continuity(self, chunks: list[FrvWireChunk]) -> None:
        expected = 0
        for chunk in chunks:
            for row in range(chunk.point_count):
                index = int(chunk.original_index[row])
                if index < 0:
                    raise _build_wire_error("INDEX_OUT_OF_RANGE")
                if index != expected:
                    if index < expected:
                        raise _build_wire_error("INDEX_DUPLICATE")
                    raise _build_wire_error("INDEX_OUT_OF_RANGE")
                expected += 1
        if expected != self._source_point_count:
            raise _build_wire_error("SOURCE_PASS_INCOMPLETE")

    def _check_source_result_matches(self, result: FrvSourceProviderResult) -> None:
        expected = self._source_result
        if result.proof.to_dict() != expected.proof.to_dict():
            raise _build_wire_error("DIGEST_MISMATCH")
        if result.source_identity != expected.source_identity:
            raise _build_wire_error("DIGEST_MISMATCH")
        if (
            to_frv_a2_provider_binding_evidence(result).binding_digest
            != to_frv_a2_provider_binding_evidence(expected).binding_digest
        ):
            raise _build_wire_error("DIGEST_MISMATCH")

    def _build_proof(self) -> FrvWireProviderProof:
        source_proof = self._source_result.proof
        manifest = self._authority_reference.manifest
        source_proof_digest = frv_digest(
            {
                "digest_domain": _FRV_WIRE_SOURCE_PROOF_DIGEST_DOMAIN,
                "source_proof": source_proof.to_dict(),
            }
        )
        source_binding_digest = to_frv_a2_provider_binding_evidence(
            self._source_result
        ).binding_digest
        components = tuple(
            sorted(
                (self._authority_reference.authority_manifest_component,)
                + manifest.w1_artifact_components,
                key=_component_sort_key,
            )
        )
        artifact_reference_id = _derive_artifact_reference_id(
            FrvA2ProviderRole.WIRE, FRV_WIRE_PROVIDER_KEY, components
        )
        assert self._wire_point_content_digest is not None
        if self._has_in_domain:
            minimum = self._min_distance
            maximum = self._max_distance
        else:
            minimum = 0.0
            maximum = 0.0
        proof = FrvWireProviderProof.__new__(FrvWireProviderProof)
        object.__setattr__(proof, "schema_version", FRV_WIRE_PROOF_SCHEMA_VERSION)
        object.__setattr__(proof, "provider_key", FRV_WIRE_PROVIDER_KEY)
        object.__setattr__(proof, "provider_role", FrvA2ProviderRole.WIRE)
        object.__setattr__(
            proof, "interface_version", FRV_WIRE_PROVIDER_INTERFACE_VERSION
        )
        object.__setattr__(proof, "algorithm_version", FRV_WIRE_ALGORITHM_VERSION)
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
        object.__setattr__(proof, "authority_mode", manifest.authority_mode)
        object.__setattr__(
            proof, "authority_manifest_digest", manifest.manifest_digest
        )
        object.__setattr__(
            proof, "wire_geometry_content_digest", self._geometry_digest
        )
        object.__setattr__(
            proof, "wire_point_content_digest", self._wire_point_content_digest
        )
        object.__setattr__(proof, "approved_wire_count", self._approved_wire_count)
        object.__setattr__(
            proof, "wire_relation_valid_count", self._relation_valid_count
        )
        object.__setattr__(
            proof, "wire_not_applicable_proven_count", self._not_applicable_count
        )
        object.__setattr__(
            proof, "outside_influence_domain_count", self._outside_count
        )
        object.__setattr__(proof, "wire_body_possible_count", self._body_count)
        object.__setattr__(proof, "wire_proximity_only_count", self._proximity_count)
        object.__setattr__(
            proof, "wire_vegetation_conflict_count", self._conflict_count
        )
        object.__setattr__(
            proof, "minimum_centerline_distance_m", _normalize_zero(minimum)
        )
        object.__setattr__(
            proof, "maximum_centerline_distance_m", _normalize_zero(maximum)
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


def load_frv_wire(
    reference: FrvSourceProviderReference,
    source_result: FrvSourceProviderResult,
    authority_reference: FrvWireAuthorityReference,
    *,
    frv_configuration: FrvConfiguration,
    provider_config: FrvWireProviderConfig,
    source_chunk_size: int = FRV_SOURCE_DEFAULT_CHUNK_SIZE,
) -> FrvWireLoadSession:
    """Open a single-use one-source-pass P-WIRE session (private implementation)."""

    if type(reference) is not FrvSourceProviderReference:
        raise TypeError("reference must be a FrvSourceProviderReference")
    if type(source_result) is not FrvSourceProviderResult:
        raise TypeError("source_result must be a FrvSourceProviderResult")
    if type(authority_reference) is not FrvWireAuthorityReference:
        raise TypeError("authority_reference must be a FrvWireAuthorityReference")
    if type(frv_configuration) is not FrvConfiguration:
        raise TypeError("frv_configuration must be a FrvConfiguration")
    if type(provider_config) is not FrvWireProviderConfig:
        raise TypeError("provider_config must be a FrvWireProviderConfig")
    if type(source_chunk_size) is not int:
        raise TypeError("source_chunk_size must be a built-in int")
    if source_chunk_size <= 0:
        raise ValueError("source_chunk_size must be positive")
    return _FrvWireLoadSession(
        reference,
        source_result,
        authority_reference,
        frv_configuration,
        provider_config,
        source_chunk_size,
    )
