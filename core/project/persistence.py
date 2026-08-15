"""Strict, copy-on-write P2A project snapshot persistence.

P2A deliberately exposes only one-shot creation and immutable reopening.  The
generation publication primitive is private so a later locking/session phase
can become the sole authoritative writer boundary.
"""

from __future__ import annotations

import hashlib
import errno
import json
import os
import re
import secrets
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from core.contracts._json import json_safe
from core.contracts.frv import (
    SourceIdentity,
    canonical_frv_json_bytes,
    frv_digest,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.project import (
    AnalysisStage,
    AnalysisStatus,
    AnalysisTreeObservationIdentity,
    ArtifactReference,
    COORDINATE_CONTEXT_SCHEMA_VERSION,
    CoordinateContext,
    CoordinateFrame,
    CoordinateUnit,
    PREDICTION_ARTIFACT_REFERENCE_SCHEMA_VERSION,
    PREDICTION_CAPABILITY_SCHEMA_VERSION,
    PREDICTION_PROVIDER_SCHEMA_VERSION,
    PREDICTION_REQUEST_SCHEMA_VERSION,
    PREDICTION_RESULT_SCHEMA_VERSION,
    PROJECT_SCHEMA_VERSION,
    PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
    PredictionArtifactReference,
    PredictionCapability,
    PredictionCapabilityState,
    PredictionExecutionMode,
    PredictionProviderDescriptor,
    PredictionRequest,
    PredictionResult,
    PredictionResultEntry,
    ProjectContract,
    ProjectSourceReference,
    ProviderPayload,
    SPAN_ANALYSIS_SCHEMA_VERSION,
    SPAN_GEOMETRY_SCHEMA_VERSION,
    SPAN_SCHEMA_VERSION,
    SpanAnalysis,
    SpanContract,
    SpanDependencySnapshot,
    SpanGeometry,
    SpanTopologyState,
    SourceIdentityBinding,
    TOWER_REFINEMENT_SCHEMA_VERSION,
    TOWER_SCHEMA_VERSION,
    TowerContract,
    TowerGeometry,
    TowerLifecycleState,
    TowerRefinement,
    TowerRefinementQuality,
    TowerRefinementStatus,
    TowerRevisionReference,
    VIEWER_REFERENCE_SCHEMA_VERSION,
    ViewerCacheState,
    ViewerReference,
    _safe_relative_posix_path,
    span_display_id_for_sequence,
    tower_display_id_for_sequence,
)
from core.contracts.tower_candidates import (
    TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION,
    TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION,
    TOWER_CANDIDATE_SET_SCHEMA_VERSION,
    TowerCandidateConfirmation,
    TowerCandidateReviewLedger,
    TowerCandidateSet,
    validate_tower_candidate_confirmation,
    validate_tower_candidate_review_ledger,
)


ACTIVE_POINTER_SCHEMA_VERSION = "treeline-project-active-pointer-v1"
GENERATION_SCHEMA_VERSION = "treeline-project-generation-v1"
GENERATION_SCHEMA_VERSION_V2 = "treeline-project-generation-v2"
SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION = (
    "treeline-project-source-reference-persistence-v1"
)
TOWER_REGISTRY_SCHEMA_VERSION = "treeline-tower-registry-v1"
SPAN_REGISTRY_SCHEMA_VERSION = "treeline-span-registry-v1"
RUN_INDEX_SCHEMA_VERSION = "treeline-run-index-v1"
CANDIDATE_STORE_INDEX_SCHEMA_VERSION = "treeline-tower-candidate-store-index-v1"

_GENERATION_RE = re.compile(r"g-[0-9]{16}-[a-z0-9]{8}\Z")
_TOWER_ID_RE = re.compile(r"T([0-9]{3,})\Z")
_SPAN_ID_RE = re.compile(r"S([0-9]{3,})\Z")
_RUN_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_CANDIDATE_ID_RE = re.compile(r"P([0-9]{3})\Z")
_STAGING_PREFIX = ".staging-"
_GENERATION_FILES = (
    "project.json",
    "source/source-reference.json",
    "towers/towers.json",
    "spans/spans.json",
    "runs/run-index.json",
)
_PROJECT_REGISTRY_REFERENCES = {
    "tower_registry_reference": "towers/towers.json",
    "span_registry_reference": "spans/spans.json",
    "run_index_reference": "runs/run-index.json",
}
_SOURCE_POLICY = "REFERENCE_ORIGINAL_SOURCE_READ_ONLY"


class PersistenceError(ValueError):
    """Base error for strict P2A persistence failures."""


class ProjectIntegrityError(PersistenceError):
    """Raised when an authoritative project snapshot is malformed or mixed."""


class FileIntegrityError(ProjectIntegrityError):
    """Raised when a published file does not match its sealed record."""


class DurabilityError(PersistenceError):
    """Raised when a supported file/directory durability primitive fails."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        path: Path,
        pointer_replaced: bool = False,
    ) -> None:
        super().__init__(message)
        self.operation = operation
        self.path = path
        self.pointer_replaced = pointer_replaced


class FileDurabilityError(DurabilityError):
    """Raised when an authoritative file cannot be durably written."""


class DirectoryDurabilityError(DurabilityError):
    """Raised for a real directory open/fsync/close failure."""


class GenerationPublicationError(DurabilityError):
    """Raised when staging-to-final generation publication fails."""


class RootPointerReplaceError(DurabilityError):
    """Raised when root-pointer replacement fails before it can be known active."""


class PostPointerDurabilityError(DurabilityError):
    """Raised when root-directory durability fails after pointer replacement."""


@dataclass(frozen=True, slots=True)
class DirectoryFsyncResult:
    """Explicit result for a directory fsync capability/operation."""

    path: Path
    supported: bool
    detail: str


_UNSUPPORTED_DIRECTORY_ERRNOS = frozenset(
    errno_value
    for errno_value in (
        getattr(errno, "EINVAL", None),
        getattr(errno, "ENOSYS", None),
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if errno_value is not None
)


def _directory_fsync_supported() -> bool:
    """Return whether CPython exposes directory fsync on this platform.

    The Windows CRT-backed ``os.open`` API does not provide a directory handle
    suitable for ``os.fsync``.  Windows therefore reports an explicit
    unsupported capability; it is not treated as a successful flush.  POSIX
    platforms attempt the actual operation and classify only documented
    unsupported errno values as a limitation.
    """

    return os.name != "nt"


def _directory_fsync_unsupported(error: OSError) -> bool:
    return error.errno in _UNSUPPORTED_DIRECTORY_ERRNOS


def _fsync_directory(path: Path) -> DirectoryFsyncResult:
    """Durably flush a directory or explicitly report platform limitation.

    Permission, path, descriptor, and other real I/O failures propagate as a
    ``DirectoryDurabilityError``.  Only an explicitly unsupported platform or
    the narrow errno set returned by a directory-fsync operation is reported
    as unsupported.
    """

    if not _directory_fsync_supported():
        return DirectoryFsyncResult(
            path=path,
            supported=False,
            detail="directory fsync is unsupported by this Python platform",
        )

    descriptor: int | None = None
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            if _directory_fsync_unsupported(exc):
                return DirectoryFsyncResult(
                    path=path,
                    supported=False,
                    detail=f"directory fsync is unsupported: errno {exc.errno}",
                )
            raise DirectoryDurabilityError(
                f"failed to open directory for durability: {path}",
                operation="directory-open",
                path=path,
            ) from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if _directory_fsync_unsupported(exc):
                return DirectoryFsyncResult(
                    path=path,
                    supported=False,
                    detail=f"directory fsync is unsupported: errno {exc.errno}",
                )
            raise DirectoryDurabilityError(
                f"failed to fsync directory: {path}",
                operation="directory-fsync",
                path=path,
            ) from exc
        except NotImplementedError:
            return DirectoryFsyncResult(
                path=path,
                supported=False,
                detail="directory fsync is unsupported by this Python platform",
            )
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise DirectoryDurabilityError(
                    f"failed to close directory durability handle: {path}",
                    operation="directory-close",
                    path=path,
                ) from exc
    return DirectoryFsyncResult(path=path, supported=True, detail="directory fsync completed")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository-approved canonical JSON bytes without a final LF."""

    return canonical_frv_json_bytes(value)


def published_json_bytes(value: Any) -> bytes:
    """Return canonical JSON bytes with exactly one final LF."""

    return canonical_json_bytes(value) + b"\n"


def logical_json_digest(value: Any) -> str:
    """Hash canonical JSON bytes without its publication-only final LF."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(data: bytes) -> str:
    """Hash exact published file bytes."""

    return hashlib.sha256(data).hexdigest()


def published_json_digest(value: Any) -> str:
    """Hash the exact canonical JSON bytes that would be published."""

    return file_sha256(published_json_bytes(value))


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PersistenceError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise PersistenceError(f"{name} keys must be strings")
    actual = set(value)
    unknown = sorted(actual - fields)
    if unknown:
        raise PersistenceError(f"{name} has unknown field(s): {', '.join(unknown)}")
    missing = sorted(fields - actual)
    if missing:
        raise PersistenceError(f"{name} is missing required field(s): {', '.join(missing)}")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise PersistenceError(f"{name} must be a JSON array")
    return list(value)


def _enum(value: Any, enum_type: type, name: str) -> Any:
    if not isinstance(value, str):
        raise PersistenceError(f"{name} must be a serialized enum value")
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"{name} has an invalid enum value") from exc


def _timestamp(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise PersistenceError(f"{name} must be a timestamp string")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00") if value.endswith("Z") else None
    except ValueError as exc:
        raise PersistenceError(f"{name} must be a valid ISO-8601 UTC timestamp") from exc
    if not value.endswith("Z") or "T" not in value:
        raise PersistenceError(f"{name} must be a valid ISO-8601 UTC timestamp")
    return value


def _bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise PersistenceError(f"{name} must be a boolean")
    return value


def _json_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PersistenceError(f"{name} must be a JSON object")
    try:
        safe = json_safe(value, name)
    except ValueError as exc:
        raise PersistenceError(str(exc)) from exc
    if not isinstance(safe, dict):
        raise PersistenceError(f"{name} must be a JSON object")
    return safe


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _decode_coordinate_context(value: Any) -> CoordinateContext:
    data = _object(
        value,
        "CoordinateContext",
        {
            "schema_version",
            "frame",
            "units",
            "origin_xyz",
            "source_content_identity_digest",
            "source_identity_record_digest",
        },
    )
    try:
        return CoordinateContext(
            schema_version=data["schema_version"],
            frame=_enum(data["frame"], CoordinateFrame, "frame"),
            units=_enum(data["units"], CoordinateUnit, "units"),
            origin_xyz=data["origin_xyz"],
            source_content_identity_digest=data["source_content_identity_digest"],
            source_identity_record_digest=data["source_identity_record_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid CoordinateContext: {exc}") from exc


def _decode_source_binding(value: Any) -> SourceIdentityBinding:
    data = _object(
        value,
        "SourceIdentityBinding",
        {
            "source_content_identity_digest",
            "source_identity_record_digest",
            "coordinate_context",
        },
    )
    try:
        return SourceIdentityBinding(
            source_content_identity_digest=data["source_content_identity_digest"],
            source_identity_record_digest=data["source_identity_record_digest"],
            coordinate_context=_decode_coordinate_context(data["coordinate_context"]),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid SourceIdentityBinding: {exc}") from exc


def _decode_source_reference(value: Any) -> ProjectSourceReference:
    data = _object(
        value,
        "ProjectSourceReference",
        {"schema_version", "source_identity", "coordinate_context"},
    )
    try:
        source_identity = SourceIdentity.from_dict(
            _object(
                data["source_identity"],
                "SourceIdentity",
                set(SourceIdentity.__dataclass_fields__),
            )
        )
        return ProjectSourceReference(
            schema_version=data["schema_version"],
            source_identity=source_identity,
            coordinate_context=_decode_coordinate_context(data["coordinate_context"]),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid ProjectSourceReference: {exc}") from exc


def _decode_viewer(value: Any) -> ViewerReference:
    data = _object(value, "ViewerReference", {"schema_version", "state", "active_cache_id"})
    try:
        return ViewerReference(
            schema_version=data["schema_version"],
            state=_enum(data["state"], ViewerCacheState, "state"),
            active_cache_id=data["active_cache_id"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid ViewerReference: {exc}") from exc


def _decode_prediction_capability(value: Any) -> PredictionCapability:
    data = _object(
        value,
        "PredictionCapability",
        {
            "schema_version",
            "status",
            "provider_id",
            "provider_version",
            "supported_input_schema_versions",
            "supported_result_schema_versions",
            "supported_execution_modes",
            "diagnostic",
        },
    )
    modes = [
        _enum(item, PredictionExecutionMode, "supported_execution_modes[]")
        for item in _sequence(data["supported_execution_modes"], "supported_execution_modes")
    ]
    try:
        return PredictionCapability(
            schema_version=data["schema_version"],
            status=_enum(data["status"], PredictionCapabilityState, "status"),
            provider_id=data["provider_id"],
            provider_version=data["provider_version"],
            supported_input_schema_versions=tuple(
                _require_strings(
                    data["supported_input_schema_versions"],
                    "supported_input_schema_versions",
                )
            ),
            supported_result_schema_versions=tuple(
                _require_strings(
                    data["supported_result_schema_versions"],
                    "supported_result_schema_versions",
                )
            ),
            supported_execution_modes=tuple(modes),
            diagnostic=data["diagnostic"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid PredictionCapability: {exc}") from exc


def _require_strings(value: Any, name: str) -> list[str]:
    values = _sequence(value, name)
    if any(type(item) is not str for item in values):
        raise PersistenceError(f"{name} must contain only strings")
    return values


def _decode_artifact_reference(value: Any) -> ArtifactReference:
    data = _object(
        value,
        "ArtifactReference",
        {"relative_path", "role", "sha256", "byte_size", "schema_version"},
    )
    try:
        return ArtifactReference(**data)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid ArtifactReference: {exc}") from exc


def _decode_prediction_artifact_reference(value: Any) -> PredictionArtifactReference:
    data = _object(
        value,
        "PredictionArtifactReference",
        {
            "schema_version",
            "relative_path",
            "role",
            "sha256",
            "byte_size",
            "artifact_schema_version",
        },
    )
    try:
        return PredictionArtifactReference(**data)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid PredictionArtifactReference: {exc}") from exc


def _decode_project(data: Any) -> ProjectContract:
    values = _object(
        data,
        "ProjectContract",
        {
            "schema_version",
            "project_id",
            "project_name",
            "created_at",
            "updated_at",
            "source_reference",
            "viewer",
            "tower_registry_reference",
            "span_registry_reference",
            "run_index_reference",
            "prediction_capability",
            "next_tower_sequence",
            "next_span_sequence",
        },
    )
    try:
        return ProjectContract(
            schema_version=values["schema_version"],
            project_id=values["project_id"],
            project_name=values["project_name"],
            created_at=values["created_at"],
            updated_at=values["updated_at"],
            source_reference=_decode_source_reference(values["source_reference"]),
            viewer=_decode_viewer(values["viewer"]),
            tower_registry_reference=values["tower_registry_reference"],
            span_registry_reference=values["span_registry_reference"],
            run_index_reference=values["run_index_reference"],
            next_tower_sequence=values["next_tower_sequence"],
            next_span_sequence=values["next_span_sequence"],
            prediction_capability=_decode_prediction_capability(
                values["prediction_capability"]
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid ProjectContract: {exc}") from exc


def project_from_dict(data: Mapping[str, Any]) -> ProjectContract:
    """Strictly reconstruct one exact frozen P1 ``ProjectContract``."""

    return _decode_project(data)


deserialize_project = project_from_dict


def _decode_tower_revision_reference(value: Any) -> TowerRevisionReference:
    data = _object(value, "TowerRevisionReference", {"tower_id", "revision"})
    try:
        return TowerRevisionReference(**data)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid TowerRevisionReference: {exc}") from exc


def _decode_tower_geometry(value: Any) -> TowerGeometry:
    data = _object(
        value,
        "TowerGeometry",
        {"base_xyz_absolute_m", "top_xyz_absolute_m", "height_m"},
    )
    try:
        return TowerGeometry(**data)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid TowerGeometry: {exc}") from exc


def _decode_tower_refinement(value: Any) -> TowerRefinement:
    data = _object(
        value,
        "TowerRefinement",
        {
            "schema_version",
            "status",
            "quality",
            "evidence",
            "algorithm_version",
            "confirmed_at",
            "local_region_reference",
        },
    )
    try:
        return TowerRefinement(
            schema_version=data["schema_version"],
            status=_enum(data["status"], TowerRefinementStatus, "status"),
            quality=_enum(data["quality"], TowerRefinementQuality, "quality"),
            evidence=_json_mapping(data["evidence"], "evidence"),
            algorithm_version=data["algorithm_version"],
            confirmed_at=data["confirmed_at"],
            local_region_reference=(
                _decode_artifact_reference(data["local_region_reference"])
                if data["local_region_reference"] is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid TowerRefinement: {exc}") from exc


def _decode_tower(value: Any) -> TowerContract:
    data = _object(
        value,
        "TowerContract",
        {
            "schema_version",
            "tower_id",
            "display_name",
            "state",
            "source_binding",
            "coordinate_context",
            "approximate_click_xyz_absolute_m",
            "refined_geometry",
            "refinement",
            "revision",
            "notes",
            "created_at",
            "updated_at",
        },
    )
    try:
        return TowerContract(
            schema_version=data["schema_version"],
            tower_id=data["tower_id"],
            display_name=data["display_name"],
            state=_enum(data["state"], TowerLifecycleState, "state"),
            source_binding=_decode_source_binding(data["source_binding"]),
            coordinate_context=_decode_coordinate_context(data["coordinate_context"]),
            approximate_click_xyz_absolute_m=data["approximate_click_xyz_absolute_m"],
            refined_geometry=_decode_tower_geometry(data["refined_geometry"]),
            refinement=_decode_tower_refinement(data["refinement"]),
            revision=data["revision"],
            notes=data["notes"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid TowerContract: {exc}") from exc


def _decode_span_geometry(value: Any) -> SpanGeometry:
    data = _object(
        value,
        "SpanGeometry",
        {
            "schema_version",
            "definition_revision",
            "coordinate_context",
            "length_m",
            "direction_xy",
            "core_bounds_min_xyz",
            "core_bounds_max_xyz",
            "halo_bounds_min_xyz",
            "halo_bounds_max_xyz",
            "configuration_digest",
            "geometry_digest",
        },
    )
    try:
        return SpanGeometry(
            schema_version=data["schema_version"],
            definition_revision=data["definition_revision"],
            coordinate_context=_decode_coordinate_context(data["coordinate_context"]),
            length_m=data["length_m"],
            direction_xy=data["direction_xy"],
            core_bounds_min_xyz=data["core_bounds_min_xyz"],
            core_bounds_max_xyz=data["core_bounds_max_xyz"],
            halo_bounds_min_xyz=data["halo_bounds_min_xyz"],
            halo_bounds_max_xyz=data["halo_bounds_max_xyz"],
            configuration_digest=data["configuration_digest"],
            geometry_digest=data["geometry_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid SpanGeometry: {exc}") from exc


def _decode_span_dependencies(value: Any) -> SpanDependencySnapshot:
    data = _object(
        value,
        "SpanDependencySnapshot",
        {
            "source_binding",
            "from_tower",
            "to_tower",
            "geometry_digest",
            "configuration_digest",
        },
    )
    try:
        return SpanDependencySnapshot(
            source_binding=_decode_source_binding(data["source_binding"]),
            from_tower=_decode_tower_revision_reference(data["from_tower"]),
            to_tower=_decode_tower_revision_reference(data["to_tower"]),
            geometry_digest=data["geometry_digest"],
            configuration_digest=data["configuration_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid SpanDependencySnapshot: {exc}") from exc


def _decode_span_analysis(value: Any) -> SpanAnalysis:
    data = _object(
        value,
        "SpanAnalysis",
        {
            "schema_version",
            "state",
            "active_run_id",
            "stage_statuses",
            "run_dependencies",
            "historical_run_ids",
        },
    )
    statuses = _object(
        data["stage_statuses"],
        "stage_statuses",
        {stage.value for stage in AnalysisStage},
    )
    stage_statuses: dict[AnalysisStage, AnalysisStatus] = {
        stage: _enum(statuses[stage.value], AnalysisStatus, f"stage_statuses.{stage.value}")
        for stage in AnalysisStage
    }
    try:
        return SpanAnalysis(
            schema_version=data["schema_version"],
            state=_enum(data["state"], AnalysisStatus, "state"),
            active_run_id=data["active_run_id"],
            stage_statuses=stage_statuses,
            run_dependencies=(
                _decode_span_dependencies(data["run_dependencies"])
                if data["run_dependencies"] is not None
                else None
            ),
            historical_run_ids=tuple(
                _require_strings(data["historical_run_ids"], "historical_run_ids")
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid SpanAnalysis: {exc}") from exc


def _decode_span(value: Any) -> SpanContract:
    data = _object(
        value,
        "SpanContract",
        {
            "schema_version",
            "span_id",
            "display_name",
            "from_tower_id",
            "to_tower_id",
            "from_tower_revision",
            "to_tower_revision",
            "source_binding",
            "topology_state",
            "geometry",
            "analysis",
            "artifact_index_reference",
            "prediction",
        },
    )
    try:
        return SpanContract(
            schema_version=data["schema_version"],
            span_id=data["span_id"],
            display_name=data["display_name"],
            from_tower_id=data["from_tower_id"],
            to_tower_id=data["to_tower_id"],
            from_tower_revision=data["from_tower_revision"],
            to_tower_revision=data["to_tower_revision"],
            source_binding=_decode_source_binding(data["source_binding"]),
            topology_state=_enum(data["topology_state"], SpanTopologyState, "topology_state"),
            geometry=_decode_span_geometry(data["geometry"]),
            analysis=_decode_span_analysis(data["analysis"]),
            artifact_index_reference=(
                _decode_artifact_reference(data["artifact_index_reference"])
                if data["artifact_index_reference"] is not None
                else None
            ),
            prediction=_decode_prediction_capability(data["prediction"]),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid SpanContract: {exc}") from exc


def _decode_provider_payload(value: Any) -> ProviderPayload:
    data = _object(value, "ProviderPayload", {"namespace", "schema_version", "data"})
    try:
        return ProviderPayload(
            namespace=data["namespace"],
            schema_version=data["schema_version"],
            data=_json_mapping(data["data"], "data"),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid ProviderPayload: {exc}") from exc


def _decode_observation(value: Any) -> AnalysisTreeObservationIdentity:
    data = _object(
        value,
        "AnalysisTreeObservationIdentity",
        {"analysis_run_id", "local_tree_id", "observation_id"},
    )
    try:
        return AnalysisTreeObservationIdentity(**data)
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid AnalysisTreeObservationIdentity: {exc}") from exc


def _decode_provider(value: Any) -> PredictionProviderDescriptor:
    data = _object(
        value,
        "PredictionProviderDescriptor",
        {
            "schema_version",
            "provider_id",
            "provider_version",
            "supported_input_schema_versions",
            "supported_result_schema_versions",
            "supported_execution_modes",
        },
    )
    try:
        return PredictionProviderDescriptor(
            schema_version=data["schema_version"],
            provider_id=data["provider_id"],
            provider_version=data["provider_version"],
            supported_input_schema_versions=tuple(
                _require_strings(
                    data["supported_input_schema_versions"],
                    "supported_input_schema_versions",
                )
            ),
            supported_result_schema_versions=tuple(
                _require_strings(
                    data["supported_result_schema_versions"],
                    "supported_result_schema_versions",
                )
            ),
            supported_execution_modes=tuple(
                _enum(item, PredictionExecutionMode, "supported_execution_modes[]")
                for item in _sequence(
                    data["supported_execution_modes"], "supported_execution_modes"
                )
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid PredictionProviderDescriptor: {exc}") from exc


def _decode_prediction_request(value: Any) -> PredictionRequest:
    data = _object(
        value,
        "PredictionRequest",
        {
            "schema_version",
            "request_id",
            "project_id",
            "source_binding",
            "span_id",
            "span_dependencies",
            "analysis_run_id",
            "static_result_schema_version",
            "coordinate_context",
            "provider",
            "tree_observations",
            "static_artifacts",
            "provider_payload",
            "requested_at",
        },
    )
    try:
        return PredictionRequest(
            schema_version=data["schema_version"],
            request_id=data["request_id"],
            project_id=data["project_id"],
            source_binding=_decode_source_binding(data["source_binding"]),
            span_id=data["span_id"],
            span_dependencies=_decode_span_dependencies(data["span_dependencies"]),
            analysis_run_id=data["analysis_run_id"],
            static_result_schema_version=data["static_result_schema_version"],
            coordinate_context=_decode_coordinate_context(data["coordinate_context"]),
            provider=_decode_provider(data["provider"]),
            tree_observations=tuple(
                _decode_observation(item)
                for item in _sequence(data["tree_observations"], "tree_observations")
            ),
            static_artifacts=tuple(
                _decode_prediction_artifact_reference(item)
                for item in _sequence(data["static_artifacts"], "static_artifacts")
            ),
            provider_payload=_decode_provider_payload(data["provider_payload"]),
            requested_at=data["requested_at"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid PredictionRequest: {exc}") from exc


def _decode_prediction_result_entry(value: Any) -> PredictionResultEntry:
    data = _object(
        value,
        "PredictionResultEntry",
        {"observation", "provider_payload", "artifacts"},
    )
    try:
        return PredictionResultEntry(
            observation=_decode_observation(data["observation"]),
            provider_payload=_decode_provider_payload(data["provider_payload"]),
            artifacts=tuple(
                _decode_prediction_artifact_reference(item)
                for item in _sequence(data["artifacts"], "artifacts")
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid PredictionResultEntry: {exc}") from exc


def _decode_prediction_result(value: Any) -> PredictionResult:
    data = _object(
        value,
        "PredictionResult",
        {
            "schema_version",
            "result_id",
            "request_id",
            "request_schema_version",
            "project_id",
            "source_binding",
            "span_id",
            "span_dependencies",
            "analysis_run_id",
            "static_result_schema_version",
            "coordinate_context",
            "provider_id",
            "provider_version",
            "model_version",
            "entries",
            "artifacts",
            "provider_payload",
            "completed_at",
        },
    )
    entries_data = _object(data["entries"], "entries", set(data["entries"]))
    try:
        return PredictionResult(
            schema_version=data["schema_version"],
            result_id=data["result_id"],
            request_id=data["request_id"],
            request_schema_version=data["request_schema_version"],
            project_id=data["project_id"],
            source_binding=_decode_source_binding(data["source_binding"]),
            span_id=data["span_id"],
            span_dependencies=_decode_span_dependencies(data["span_dependencies"]),
            analysis_run_id=data["analysis_run_id"],
            static_result_schema_version=data["static_result_schema_version"],
            coordinate_context=_decode_coordinate_context(data["coordinate_context"]),
            provider_id=data["provider_id"],
            provider_version=data["provider_version"],
            model_version=data["model_version"],
            entries={
                key: _decode_prediction_result_entry(item)
                for key, item in entries_data.items()
            },
            artifacts=tuple(
                _decode_prediction_artifact_reference(item)
                for item in _sequence(data["artifacts"], "artifacts")
            ),
            provider_payload=_decode_provider_payload(data["provider_payload"]),
            completed_at=data["completed_at"],
        )
    except (TypeError, ValueError) as exc:
        raise PersistenceError(f"invalid PredictionResult: {exc}") from exc


def _validate_source_history(
    current: SourceIdentity,
    history: Sequence[SourceIdentity],
    *,
    history_name: str,
) -> None:
    """Freeze the current + history source access-record invariants.

    Every accepted access record for one project must describe the same frozen
    scientific content identity, must be internally digest-consistent per the
    reviewed FRV contract, and must carry a unique ``source_identity_record_digest``.
    The current record digest appears exactly once as CURRENT and never in history,
    so no record digest can resolve ambiguously to more than one access record.
    """
    validate_source_content_identity_digest(current)
    validate_source_identity_record_digest(current)
    expected_content_digest = current.source_content_identity_digest
    current_record_digest = current.source_identity_record_digest
    seen_record_digests: set[str] = set()
    for identity in history:
        if type(identity) is not SourceIdentity:
            raise PersistenceError(f"{history_name} must contain SourceIdentity values")
        validate_source_content_identity_digest(identity)
        validate_source_identity_record_digest(identity)
        if identity.source_content_identity_digest != expected_content_digest:
            raise PersistenceError(
                f"{history_name} record does not match the project source content identity"
            )
        if identity.source_identity_record_digest in seen_record_digests:
            raise PersistenceError(f"{history_name} contains duplicate record digests")
        if identity.source_identity_record_digest == current_record_digest:
            raise PersistenceError(
                f"the current source record digest must not appear in {history_name}"
            )
        seen_record_digests.add(identity.source_identity_record_digest)


@dataclass(frozen=True, slots=True)
class SourceReferenceSidecar:
    """Immutable source access envelope paired with one P1 source reference."""

    schema_version: str
    generation_id: str
    project_id: str
    project_source_reference: ProjectSourceReference
    access_metadata: Mapping[str, Any]
    source_identity_history: tuple[SourceIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION:
            raise PersistenceError("unsupported source-reference persistence schema version")
        _validate_generation_id(self.generation_id)
        if self.project_source_reference.__class__ is not ProjectSourceReference:
            raise PersistenceError("project_source_reference must be a ProjectSourceReference")
        if not isinstance(self.access_metadata, Mapping):
            raise PersistenceError("access_metadata must be a mapping")
        object.__setattr__(
            self,
            "access_metadata",
            _freeze_json(_json_mapping(self.access_metadata, "access_metadata")),
        )
        history = tuple(self.source_identity_history)
        if any(type(item) is not SourceIdentity for item in history):
            raise PersistenceError("source_identity_history must contain SourceIdentity values")
        _validate_source_history(
            self.project_source_reference.source_identity,
            history,
            history_name="source_identity_history",
        )
        object.__setattr__(self, "source_identity_history", history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "project_id": self.project_id,
            "project_source_reference": self.project_source_reference.to_dict(),
            "access_metadata": _thaw_json(self.access_metadata),
            "source_identity_history": [item.to_dict() for item in self.source_identity_history],
        }


@dataclass(frozen=True, slots=True)
class TowerRegistryEntry:
    """One logical tower identity with immutable revision history."""

    tower_id: str
    current_revision: int
    revisions: tuple[TowerContract, ...]

    def __post_init__(self) -> None:
        if type(self.tower_id) is not str:
            raise PersistenceError("tower registry entry tower_id must be a string")
        matched = _TOWER_ID_RE.fullmatch(self.tower_id)
        if matched is None or tower_display_id_for_sequence(int(matched.group(1))) != self.tower_id:
            raise PersistenceError("tower registry entry tower_id is not canonical")
        revisions = tuple(self.revisions)
        if not revisions or any(type(item) is not TowerContract for item in revisions):
            raise PersistenceError("tower registry entry revisions must contain TowerContract values")
        if any(item.tower_id != self.tower_id for item in revisions):
            raise PersistenceError("tower registry entry revision tower_id does not match entry")
        revision_numbers = tuple(item.revision for item in revisions)
        if revision_numbers != tuple(sorted(revision_numbers)):
            raise PersistenceError("tower revisions must be in ascending order")
        if len(set(revision_numbers)) != len(revision_numbers):
            raise PersistenceError("tower revisions must be unique")
        if type(self.current_revision) is not int or self.current_revision <= 0:
            raise PersistenceError("tower current_revision must be positive")
        expected_revisions = tuple(range(1, self.current_revision + 1))
        if revision_numbers != expected_revisions:
            raise PersistenceError(
                "tower revisions must be the complete contiguous history 1..current_revision"
            )
        object.__setattr__(self, "revisions", revisions)

    @property
    def current(self) -> TowerContract:
        return self.revisions[-1]

    def revision(self, revision: int) -> TowerContract:
        for item in self.revisions:
            if item.revision == revision:
                return item
        raise KeyError(f"unknown tower revision: {self.tower_id} r{revision}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tower_id": self.tower_id,
            "current_revision": self.current_revision,
            "revisions": [item.to_dict() for item in self.revisions],
        }


def _freeze_registry_reservations(
    value: Sequence[Any], name: str, *, prefix: str
) -> tuple[Mapping[str, Any], ...]:
    reservations: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for item in _sequence(value, name):
        data = _object(item, f"{name}[]", {"sequence", "display_id"})
        sequence = data["sequence"]
        display_id = data["display_id"]
        if type(sequence) is not int or sequence <= 0:
            raise PersistenceError(f"{name} sequence must be a positive integer")
        if type(display_id) is not str or display_id != f"{prefix}{sequence:03d}":
            raise PersistenceError(f"{name} display_id does not match sequence")
        if sequence in seen:
            raise PersistenceError(f"{name} contains duplicate sequences")
        seen.add(sequence)
        reservations.append(MappingProxyType({"sequence": sequence, "display_id": display_id}))
    reservations.sort(key=lambda item: item["sequence"])
    return tuple(reservations)


@dataclass(frozen=True, slots=True)
class TowerRegistry:
    """Immutable Tower registry envelope with identity reservations and history."""

    schema_version: str
    generation_id: str
    project_id: str
    reserved_sequences: tuple[Mapping[str, Any], ...] = ()
    records: tuple[TowerRegistryEntry, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != TOWER_REGISTRY_SCHEMA_VERSION:
            raise PersistenceError("unsupported tower-registry schema version")
        _validate_generation_id(self.generation_id)
        object.__setattr__(
            self,
            "reserved_sequences",
            _freeze_registry_reservations(
                self.reserved_sequences, "tower reserved_sequences", prefix="T"
            ),
        )
        records = tuple(self.records)
        if any(type(item) is not TowerRegistryEntry for item in records):
            raise PersistenceError("tower registry records must contain TowerRegistryEntry values")
        tower_ids = tuple(item.tower_id for item in records)
        if len(set(tower_ids)) != len(tower_ids):
            raise PersistenceError("tower registry contains duplicate tower identities")
        if tower_ids != tuple(sorted(tower_ids, key=lambda value: int(value[1:]))):
            raise PersistenceError("tower registry records must be in sequence order")
        object.__setattr__(self, "records", records)

    def get(self, tower_id: str) -> TowerRegistryEntry:
        for item in self.records:
            if item.tower_id == tower_id:
                return item
        raise KeyError(tower_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "project_id": self.project_id,
            "reserved_sequences": [_thaw_json(item) for item in self.reserved_sequences],
            "records": [item.to_dict() for item in self.records],
        }


@dataclass(frozen=True, slots=True)
class SpanRegistry:
    """Immutable one-record-per-span topology registry."""

    schema_version: str
    generation_id: str
    project_id: str
    reserved_sequences: tuple[Mapping[str, Any], ...] = ()
    records: tuple[SpanContract, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != SPAN_REGISTRY_SCHEMA_VERSION:
            raise PersistenceError("unsupported span-registry schema version")
        _validate_generation_id(self.generation_id)
        object.__setattr__(
            self,
            "reserved_sequences",
            _freeze_registry_reservations(
                self.reserved_sequences, "span reserved_sequences", prefix="S"
            ),
        )
        records = tuple(self.records)
        if any(type(item) is not SpanContract for item in records):
            raise PersistenceError("span registry records must contain SpanContract values")
        span_ids = tuple(item.span_id for item in records)
        for span_id in span_ids:
            matched = _SPAN_ID_RE.fullmatch(span_id)
            if matched is None or span_display_id_for_sequence(int(matched.group(1))) != span_id:
                raise PersistenceError("span registry contains a non-canonical span ID")
        if len(set(span_ids)) != len(span_ids):
            raise PersistenceError("span registry contains duplicate span identities")
        if span_ids != tuple(sorted(span_ids, key=lambda value: int(value[1:]))):
            raise PersistenceError("span registry records must be in sequence order")
        object.__setattr__(self, "records", records)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "project_id": self.project_id,
            "reserved_sequences": [_thaw_json(item) for item in self.reserved_sequences],
            "records": [item.to_dict() for item in self.records],
        }


@dataclass(frozen=True, slots=True)
class RunIndexEntry:
    """One validated historical analysis-run projection in the P2 run index."""

    run_id: str
    span_id: str
    status: AnalysisStatus
    artifact_root_reference: str | None
    source_content_identity_digest: str
    source_identity_record_digest: str
    span_dependencies: SpanDependencySnapshot
    created_at: str
    completed_at: str | None

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or _RUN_ID_RE.fullmatch(self.run_id) is None:
            raise PersistenceError("run_id must be a lower-case opaque run identifier")
        _id_sequence(self.span_id, _SPAN_ID_RE, "span_id")
        if type(self.status) is not AnalysisStatus or self.status not in {
            AnalysisStatus.READY,
            AnalysisStatus.RUNNING,
            AnalysisStatus.COMPLETE,
            AnalysisStatus.FAILED,
            AnalysisStatus.INTERRUPTED,
            AnalysisStatus.STALE,
        }:
            raise PersistenceError("run status is not permitted by the frozen run-index contract")
        if self.artifact_root_reference is not None:
            if type(self.artifact_root_reference) is not str:
                raise PersistenceError("artifact_root_reference must be a project-relative path or null")
            try:
                _safe_relative_posix_path(self.artifact_root_reference, "artifact_root_reference")
            except ValueError as exc:
                raise PersistenceError(f"invalid artifact_root_reference: {exc}") from exc
        _validate_digest(self.source_content_identity_digest, "source_content_identity_digest")
        _validate_digest(self.source_identity_record_digest, "source_identity_record_digest")
        if type(self.span_dependencies) is not SpanDependencySnapshot:
            raise PersistenceError("span_dependencies must be a SpanDependencySnapshot")
        created_at = _timestamp(self.created_at, "created_at")
        created = datetime.fromisoformat(created_at[:-1] + "+00:00")
        if self.completed_at is None:
            if self.status in {
                AnalysisStatus.COMPLETE,
                AnalysisStatus.FAILED,
                AnalysisStatus.INTERRUPTED,
            }:
                raise PersistenceError("terminal run status requires completed_at")
        else:
            completed_at = _timestamp(self.completed_at, "completed_at")
            completed = datetime.fromisoformat(completed_at[:-1] + "+00:00")
            if completed < created:
                raise PersistenceError("completed_at must not precede created_at")
            if self.status in {AnalysisStatus.READY, AnalysisStatus.RUNNING}:
                raise PersistenceError("nonterminal run status must not have completed_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "span_id": self.span_id,
            "status": self.status.value,
            "artifact_root_reference": self.artifact_root_reference,
            "source_content_identity_digest": self.source_content_identity_digest,
            "source_identity_record_digest": self.source_identity_record_digest,
            "span_dependencies": self.span_dependencies.to_dict(),
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass(frozen=True, slots=True)
class RunIndex:
    """Immutable, identity-keyed P2 run-index envelope."""

    schema_version: str
    generation_id: str
    project_id: str
    entries: tuple[RunIndexEntry, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != RUN_INDEX_SCHEMA_VERSION:
            raise PersistenceError("unsupported run-index schema version")
        _validate_generation_id(self.generation_id)
        entries = tuple(self.entries)
        if any(type(item) is not RunIndexEntry for item in entries):
            raise PersistenceError("run-index entries must contain RunIndexEntry values")
        run_ids = tuple(item.run_id for item in entries)
        if len(run_ids) != len(set(run_ids)):
            raise PersistenceError("run index contains duplicate run identities")
        if run_ids != tuple(sorted(run_ids)):
            raise PersistenceError("run-index entries must be sorted by run_id")
        object.__setattr__(self, "entries", entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generation_id": self.generation_id,
            "project_id": self.project_id,
            "entries": [item.to_dict() for item in self.entries],
        }


@dataclass(frozen=True, slots=True)
class CandidateStore:
    """Immutable V5A1-B candidate artifacts bound to one project generation.

    The on-disk index (``candidates/index.json``) binds these artifacts to one
    generation through digest-protected relative references; this in-memory
    envelope carries the decoded immutable contracts.  ``CandidateStore()`` is
    the empty store used by legacy v1 generations that never contain a candidate
    store.
    """

    schema_version: str = CANDIDATE_STORE_INDEX_SCHEMA_VERSION
    candidate_sets: tuple[TowerCandidateSet, ...] = ()
    review_ledgers: tuple[TowerCandidateReviewLedger, ...] = ()
    confirmations: tuple[TowerCandidateConfirmation, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != CANDIDATE_STORE_INDEX_SCHEMA_VERSION:
            raise PersistenceError("unsupported candidate-store index schema version")
        sets = tuple(self.candidate_sets)
        if any(type(item) is not TowerCandidateSet for item in sets):
            raise PersistenceError(
                "candidate store candidate_sets must contain TowerCandidateSet values"
            )
        if len({item.candidate_set_id for item in sets}) != len(sets):
            raise PersistenceError("candidate store contains duplicate candidate-set identities")
        ledgers = tuple(self.review_ledgers)
        if any(type(item) is not TowerCandidateReviewLedger for item in ledgers):
            raise PersistenceError(
                "candidate store review_ledgers must contain TowerCandidateReviewLedger values"
            )
        if len({item.review_ledger_id for item in ledgers}) != len(ledgers):
            raise PersistenceError("candidate store contains duplicate review-ledger identities")
        sets_by_id = {item.candidate_set_id: item for item in sets}
        ledger_by_set: dict[str, TowerCandidateReviewLedger] = {}
        for ledger in ledgers:
            if len(ledger.candidate_sets) != 1:
                raise PersistenceError(
                    "a persisted review ledger must declare exactly one candidate set"
                )
            reference = ledger.candidate_sets[0]
            actual = sets_by_id.get(reference.candidate_set_id)
            if actual is None or actual.reference != reference:
                raise PersistenceError(
                    "review ledger candidate-set reference does not resolve to the actual set"
                )
            if reference.candidate_set_id in ledger_by_set:
                raise PersistenceError("at most one review ledger is permitted per candidate set")
            ledger_by_set[reference.candidate_set_id] = ledger
            try:
                validate_tower_candidate_review_ledger(actual, ledger)
            except ValueError as exc:
                raise PersistenceError(f"invalid tower-candidate review ledger: {exc}") from exc
        confirmations = tuple(self.confirmations)
        if any(type(item) is not TowerCandidateConfirmation for item in confirmations):
            raise PersistenceError(
                "candidate store confirmations must contain TowerCandidateConfirmation values"
            )
        confirmed_keys: set[tuple[str, str]] = set()
        for confirmation in confirmations:
            reference = confirmation.candidate_set_reference
            actual = sets_by_id.get(reference.candidate_set_id)
            if actual is None or actual.reference != reference:
                raise PersistenceError(
                    "confirmation candidate-set reference does not resolve to the actual set"
                )
            ledger = ledger_by_set.get(reference.candidate_set_id)
            if ledger is None:
                raise PersistenceError(
                    "confirmation requires a review ledger for its candidate set"
                )
            key = (reference.candidate_set_id, confirmation.candidate_id)
            if key in confirmed_keys:
                raise PersistenceError("a candidate may have at most one confirmation")
            confirmed_keys.add(key)
            try:
                validate_tower_candidate_confirmation(actual, ledger, confirmation)
            except ValueError as exc:
                raise PersistenceError(f"invalid tower-candidate confirmation: {exc}") from exc


def _candidate_set_relative_path(candidate_set_id: str) -> str:
    return f"candidates/candidate-sets/{candidate_set_id}.json"


def _review_ledger_relative_path(review_ledger_id: str) -> str:
    return f"candidates/review-ledgers/{review_ledger_id}.json"


def _confirmation_relative_path(
    candidate_set_id: str, candidate_id: str, accepted_review_revision: int
) -> str:
    return (
        f"candidates/confirmations/{candidate_set_id}-{candidate_id}-r"
        f"{accepted_review_revision}.json"
    )


def _candidate_store_relative_paths(store: CandidateStore) -> tuple[str, ...]:
    # A v2 generation always carries its candidate index, even when the store
    # holds no candidate artifacts yet.
    paths = ["candidates/index.json"]
    paths.extend(
        _candidate_set_relative_path(item.candidate_set_id)
        for item in sorted(store.candidate_sets, key=lambda s: s.candidate_set_id)
    )
    paths.extend(
        _review_ledger_relative_path(item.review_ledger_id)
        for item in sorted(store.review_ledgers, key=lambda ledger: ledger.review_ledger_id)
    )
    paths.extend(
        _confirmation_relative_path(
            item.candidate_set_reference.candidate_set_id,
            item.candidate_id,
            item.accepted_review_revision,
        )
        for item in sorted(
            store.confirmations,
            key=lambda c: (
                c.candidate_set_reference.candidate_set_id,
                c.candidate_id,
                c.accepted_review_revision,
            ),
        )
    )
    return tuple(paths)


def _span_analysis_projection_for_run(
    entry: RunIndexEntry,
    *,
    historical_run_ids: tuple[str, ...],
) -> SpanAnalysis:
    """Build the one current Span-analysis projection for a run-index entry.

    A Span stores the current/latest run projection even after a terminal
    transition. Prior runs remain historical; terminal current runs are also
    retained in that history. P1 independently requires this for STALE.
    """

    if type(entry) is not RunIndexEntry:
        raise PersistenceError("entry must be a RunIndexEntry")
    history = tuple(historical_run_ids)
    if entry.status in {
        AnalysisStatus.COMPLETE,
        AnalysisStatus.FAILED,
        AnalysisStatus.INTERRUPTED,
        AnalysisStatus.STALE,
    } and entry.run_id not in history:
        history = (*history, entry.run_id)
    return SpanAnalysis(
        schema_version=SPAN_ANALYSIS_SCHEMA_VERSION,
        state=entry.status,
        active_run_id=entry.run_id,
        stage_statuses={stage: entry.status for stage in AnalysisStage},
        run_dependencies=entry.span_dependencies,
        historical_run_ids=history,
    )


@dataclass(frozen=True, slots=True)
class FileRecord:
    relative_path: str
    byte_size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    schema_version: str
    project_id: str
    generation_id: str
    sealed: bool
    files: tuple[FileRecord, ...]
    project_manifest_logical_digest: str
    created_at: str
    published_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "generation_id": self.generation_id,
            "sealed": self.sealed,
            "files": [item.to_dict() for item in self.files],
            "project_manifest_logical_digest": self.project_manifest_logical_digest,
            "created_at": self.created_at,
            "published_at": self.published_at,
        }


@dataclass(frozen=True, slots=True)
class LoadedProject:
    """One coherent immutable active generation and its decoded envelopes."""

    project: ProjectContract
    source_sidecar: SourceReferenceSidecar
    tower_registry: TowerRegistry
    span_registry: SpanRegistry
    run_index: RunIndex
    candidate_store: CandidateStore
    generation_id: str
    generation_manifest: GenerationManifest
    project_root: Path
    generation_path: Path


def _validate_generation_id(value: Any) -> str:
    if type(value) is not str or _GENERATION_RE.fullmatch(value) is None:
        raise PersistenceError("generation_id must be a final lower-case generation identifier")
    return value


def _validate_digest(value: Any, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise PersistenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _parse_json_bytes(data: bytes, name: str) -> Any:
    if not data.endswith(b"\n") or data.endswith(b"\n\n"):
        raise ProjectIntegrityError(f"{name} must end with exactly one LF")
    if data.startswith(b"\xef\xbb\xbf"):
        raise ProjectIntegrityError(f"{name} must not contain a UTF-8 BOM")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} is forbidden")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            data[:-1].decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProjectIntegrityError(f"{name} is not valid canonical JSON: {exc}") from exc
    try:
        if published_json_bytes(parsed) != data:
            raise ProjectIntegrityError(f"{name} is not canonical JSON")
    except (TypeError, ValueError) as exc:
        raise ProjectIntegrityError(f"{name} is not canonical JSON: {exc}") from exc
    return parsed


def _read_json(path: Path, name: str) -> Any:
    try:
        data = path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise ProjectIntegrityError(f"missing or unreadable {name}") from exc
    return _parse_json_bytes(data, name)


def _write_json(path: Path, value: Any) -> bytes:
    data = published_json_bytes(value)
    try:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise FileDurabilityError(
            f"failed to durably write file: {path}",
            operation="file-write-fsync",
            path=path,
        ) from exc
    return data


def _confined_path(base: Path, relative: str, name: str) -> Path:
    try:
        _safe_relative_posix_path(relative, name)
    except (TypeError, ValueError) as exc:
        raise ProjectIntegrityError(str(exc)) from exc
    base_resolved = base.resolve(strict=False)
    candidate = base.joinpath(*relative.split("/"))
    candidate_resolved = candidate.resolve(strict=False)
    try:
        candidate_resolved.relative_to(base_resolved)
    except ValueError as exc:
        raise ProjectIntegrityError(f"{name} escapes its allowed root") from exc
    return candidate


def _generation_parent(root: Path) -> Path:
    generations = _confined_path(root, "state/generations", "generation directory")
    generations.mkdir(parents=True, exist_ok=True)
    return generations


def _next_generation_id(parent: Path) -> str:
    maximum = 0
    for child in parent.iterdir():
        if not child.is_dir() or _GENERATION_RE.fullmatch(child.name) is None:
            continue
        maximum = max(maximum, int(child.name.split("-")[1]))
    while True:
        candidate = f"g-{maximum + 1:016d}-{secrets.token_hex(4)}"
        if not (parent / candidate).exists():
            return candidate


def _generation_path(root: Path, generation_id: str) -> Path:
    _validate_generation_id(generation_id)
    return _confined_path(
        root,
        f"state/generations/{generation_id}",
        "active generation",
    )


def _project_reference_path(
    project: ProjectContract,
    field_name: str,
    root: Path,
    generation_path: Path,
    generation_id: str,
) -> Path:
    reference = getattr(project, field_name)
    expected_local = _PROJECT_REGISTRY_REFERENCES[field_name]
    expected_full = f"state/generations/{generation_id}/{expected_local}"
    if reference == expected_local:
        return _confined_path(generation_path, expected_local, field_name)
    if reference == expected_full:
        return _confined_path(root, reference, field_name)
    raise ProjectIntegrityError(
        f"{field_name} must reference {expected_local} within the selected generation"
    )


def _decode_source_sidecar(value: Any) -> SourceReferenceSidecar:
    data = _object(
        value,
        "SourceReferenceSidecar",
        {
            "schema_version",
            "generation_id",
            "project_id",
            "project_source_reference",
            "access_metadata",
            "source_identity_history",
        },
    )
    access = _object(
        data["access_metadata"],
        "access_metadata",
        {
            "policy",
            "current_resolved_source_path",
            "file_size_bytes",
            "source_file_sha256",
            "point_count",
            "bounds_min_xyz",
            "bounds_max_xyz",
            "header",
            "source_available_at_save",
            "recorded_at",
        },
    )
    history: list[SourceIdentity] = []
    for item in _sequence(data["source_identity_history"], "source_identity_history"):
        history.append(
            SourceIdentity.from_dict(
                _object(item, "SourceIdentity", set(SourceIdentity.__dataclass_fields__))
            )
        )
    try:
        sidecar = SourceReferenceSidecar(
            schema_version=data["schema_version"],
            generation_id=data["generation_id"],
            project_id=data["project_id"],
            project_source_reference=_decode_source_reference(data["project_source_reference"]),
            access_metadata=access,
            source_identity_history=tuple(history),
        )
    except (TypeError, ValueError) as exc:
        raise ProjectIntegrityError(f"invalid source sidecar: {exc}") from exc
    _validate_access_metadata(sidecar)
    return sidecar


def _validate_access_metadata(sidecar: SourceReferenceSidecar) -> None:
    metadata = sidecar.access_metadata
    if metadata.get("policy") != _SOURCE_POLICY:
        raise ProjectIntegrityError("source sidecar has an unsupported access policy")
    path = metadata.get("current_resolved_source_path")
    if (
        type(path) is not str
        or not path
        or path != sidecar.project_source_reference.source_identity.resolved_access_path
    ):
        raise ProjectIntegrityError("source sidecar access path does not match SourceIdentity")
    if type(metadata.get("source_available_at_save")) is not bool:
        raise ProjectIntegrityError("source_available_at_save must be a boolean")
    if type(metadata.get("file_size_bytes")) is not int or metadata["file_size_bytes"] < 0:
        raise ProjectIntegrityError("file_size_bytes must be a non-negative integer")
    if type(metadata.get("point_count")) is not int or metadata["point_count"] < 0:
        raise ProjectIntegrityError("point_count must be a non-negative integer")
    if (
        metadata.get("source_file_sha256")
        != sidecar.project_source_reference.source_identity.source_file_sha256
    ):
        raise ProjectIntegrityError(
            "source sidecar source_file_sha256 does not match SourceIdentity"
        )
    if list(metadata.get("bounds_min_xyz", ())) != list(
        sidecar.project_source_reference.source_identity.source_bounds_min_xyz
    ):
        raise ProjectIntegrityError("source sidecar bounds_min_xyz does not match SourceIdentity")
    if list(metadata.get("bounds_max_xyz", ())) != list(
        sidecar.project_source_reference.source_identity.source_bounds_max_xyz
    ):
        raise ProjectIntegrityError("source sidecar bounds_max_xyz does not match SourceIdentity")
    _timestamp(metadata.get("recorded_at"), "access_metadata.recorded_at")
    if not isinstance(metadata.get("header"), Mapping):
        raise ProjectIntegrityError("access_metadata.header must be a JSON object")


def _default_access_metadata(project: ProjectContract) -> dict[str, Any]:
    identity = project.source_reference.source_identity
    path = Path(identity.resolved_access_path)
    available = False
    file_size = 0
    try:
        stat = path.stat()
    except OSError:
        pass
    else:
        available = path.is_file()
        file_size = stat.st_size if available else 0
    return {
        "policy": _SOURCE_POLICY,
        "current_resolved_source_path": identity.resolved_access_path,
        "file_size_bytes": file_size,
        "source_file_sha256": identity.source_file_sha256,
        "point_count": identity.source_point_count,
        "bounds_min_xyz": list(identity.source_bounds_min_xyz),
        "bounds_max_xyz": list(identity.source_bounds_max_xyz),
        "header": {},
        "source_available_at_save": available,
        "recorded_at": _utc_timestamp(),
    }


def _build_source_sidecar(
    project: ProjectContract,
    generation_id: str,
    source_access_metadata: Mapping[str, Any] | None,
) -> SourceReferenceSidecar:
    metadata = _default_access_metadata(project)
    if source_access_metadata is not None:
        if not isinstance(source_access_metadata, Mapping):
            raise PersistenceError("source_access_metadata must be a mapping")
        unknown = sorted(set(source_access_metadata) - set(metadata))
        if unknown:
            raise PersistenceError(
                f"source_access_metadata has unknown field(s): {', '.join(unknown)}"
            )
        metadata.update(source_access_metadata)
    sidecar = SourceReferenceSidecar(
        schema_version=SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION,
        generation_id=generation_id,
        project_id=project.project_id,
        project_source_reference=project.source_reference,
        access_metadata=metadata,
        source_identity_history=(),
    )
    _validate_access_metadata(sidecar)
    return sidecar


def _rebind_source_sidecar(
    sidecar: SourceReferenceSidecar, *, generation_id: str, project: ProjectContract
) -> SourceReferenceSidecar:
    if sidecar.project_id != project.project_id:
        raise PersistenceError("source sidecar project identity does not match candidate project")
    if sidecar.project_source_reference != project.source_reference:
        raise PersistenceError("source sidecar source reference does not match candidate project")
    return SourceReferenceSidecar(
        schema_version=SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION,
        generation_id=generation_id,
        project_id=project.project_id,
        project_source_reference=project.source_reference,
        access_metadata=sidecar.access_metadata,
        source_identity_history=sidecar.source_identity_history,
    )


def _decode_registry(
    value: Any,
    project: ProjectContract,
    generation_id: str,
) -> TowerRegistry | SpanRegistry:
    data = _object(
        value,
        "registry",
        {"schema_version", "generation_id", "project_id", "reserved_sequences", "records"},
    )
    _sequence(data["reserved_sequences"], "reserved_sequences")
    _sequence(data["records"], "records")
    if data["project_id"] != project.project_id:
        raise ProjectIntegrityError("registry project_id does not match project")
    if data["generation_id"] != generation_id:
        raise ProjectIntegrityError("registry generation_id does not match generation")
    if data["schema_version"] == TOWER_REGISTRY_SCHEMA_VERSION:
        reservations = _freeze_registry_reservations(
            data["reserved_sequences"], "tower reserved_sequences", prefix="T"
        )
        entries: list[TowerRegistryEntry] = []
        for item in _sequence(data["records"], "tower records"):
            entry = _object(
                item,
                "TowerRegistryEntry",
                {"tower_id", "current_revision", "revisions"},
            )
            revisions = tuple(
                _decode_tower(revision)
                for revision in _sequence(entry["revisions"], "tower revisions")
            )
            try:
                entries.append(
                    TowerRegistryEntry(
                        tower_id=entry["tower_id"],
                        current_revision=entry["current_revision"],
                        revisions=revisions,
                    )
                )
            except (TypeError, ValueError, PersistenceError) as exc:
                raise ProjectIntegrityError(f"invalid TowerRegistryEntry: {exc}") from exc
        return TowerRegistry(
            schema_version=data["schema_version"],
            generation_id=data["generation_id"],
            project_id=data["project_id"],
            reserved_sequences=reservations,
            records=tuple(entries),
        )
    if data["schema_version"] == SPAN_REGISTRY_SCHEMA_VERSION:
        reservations = _freeze_registry_reservations(
            data["reserved_sequences"], "span reserved_sequences", prefix="S"
        )
        records = tuple(
            _decode_span(item) for item in _sequence(data["records"], "span records")
        )
        return SpanRegistry(
            schema_version=data["schema_version"],
            generation_id=data["generation_id"],
            project_id=data["project_id"],
            reserved_sequences=reservations,
            records=records,
        )
    raise ProjectIntegrityError("unsupported registry schema version")


def _decode_run_index_entry(value: Any) -> RunIndexEntry:
    data = _object(
        value,
        "RunIndexEntry",
        {
            "run_id",
            "span_id",
            "status",
            "artifact_root_reference",
            "source_content_identity_digest",
            "source_identity_record_digest",
            "span_dependencies",
            "created_at",
            "completed_at",
        },
    )
    try:
        return RunIndexEntry(
            run_id=data["run_id"],
            span_id=data["span_id"],
            status=_enum(data["status"], AnalysisStatus, "run status"),
            artifact_root_reference=data["artifact_root_reference"],
            source_content_identity_digest=data["source_content_identity_digest"],
            source_identity_record_digest=data["source_identity_record_digest"],
            span_dependencies=_decode_span_dependencies(data["span_dependencies"]),
            created_at=data["created_at"],
            completed_at=data["completed_at"],
        )
    except (TypeError, ValueError, PersistenceError) as exc:
        raise ProjectIntegrityError(f"invalid RunIndexEntry: {exc}") from exc


def _decode_run_index(value: Any, project: ProjectContract, generation_id: str) -> RunIndex:
    data = _object(value, "RunIndex", {"schema_version", "generation_id", "project_id", "entries"})
    entries = tuple(
        _decode_run_index_entry(item)
        for item in _sequence(data["entries"], "entries")
    )
    if data["schema_version"] != RUN_INDEX_SCHEMA_VERSION:
        raise ProjectIntegrityError("unsupported run-index schema version")
    if data["project_id"] != project.project_id:
        raise ProjectIntegrityError("run index project_id does not match project")
    if data["generation_id"] != generation_id:
        raise ProjectIntegrityError("run index generation_id does not match generation")
    try:
        return RunIndex(
            schema_version=data["schema_version"],
            generation_id=data["generation_id"],
            project_id=data["project_id"],
            entries=entries,
        )
    except PersistenceError as exc:
        raise ProjectIntegrityError(f"invalid RunIndex: {exc}") from exc


def _validate_candidate_store_relative_path(
    value: Any, prefix: str, name: str
) -> str:
    if type(value) is not str or not value.startswith(prefix):
        raise ProjectIntegrityError(f"{name} must be a generation-relative path under {prefix}")
    try:
        _safe_relative_posix_path(value, name)
    except (TypeError, ValueError) as exc:
        raise ProjectIntegrityError(f"{name} is an unsafe relative path: {exc}") from exc
    return value


def _decode_candidate_store_index(
    data: Any, project: ProjectContract, generation_id: str
) -> dict[str, Any]:
    value = _object(
        data,
        "CandidateStoreIndex",
        {
            "schema_version",
            "project_id",
            "generation_id",
            "candidate_sets",
            "review_ledgers",
            "confirmations",
        },
    )
    if value["schema_version"] != CANDIDATE_STORE_INDEX_SCHEMA_VERSION:
        raise ProjectIntegrityError("unsupported candidate-store index schema version")
    if value["project_id"] != project.project_id:
        raise ProjectIntegrityError("candidate store index project_id does not match project")
    if value["generation_id"] != generation_id:
        raise ProjectIntegrityError("candidate store index generation_id does not match generation")

    set_records: list[dict[str, Any]] = []
    seen_set_ids: set[str] = set()
    for item in _sequence(value["candidate_sets"], "candidate_sets"):
        record = _object(
            item,
            "CandidateSetIndexEntry",
            {
                "candidate_set_id",
                "candidate_set_digest",
                "schema_version",
                "relative_path",
                "sha256",
                "byte_size",
            },
        )
        if type(record["candidate_set_id"]) is not str or _RUN_ID_RE.fullmatch(
            record["candidate_set_id"]
        ) is None:
            raise ProjectIntegrityError("candidate_set_id must be a lower-case opaque identifier")
        if record["candidate_set_id"] in seen_set_ids:
            raise ProjectIntegrityError("candidate store index has duplicate candidate-set ids")
        seen_set_ids.add(record["candidate_set_id"])
        _validate_digest(record["candidate_set_digest"], "candidate_set_digest")
        if record["schema_version"] != TOWER_CANDIDATE_SET_SCHEMA_VERSION:
            raise ProjectIntegrityError("unsupported candidate-set artifact schema version")
        _validate_candidate_store_relative_path(
            record["relative_path"], "candidates/candidate-sets/", "candidate-set relative_path"
        )
        _validate_digest(record["sha256"], "candidate-set artifact sha256")
        if type(record["byte_size"]) is not int or record["byte_size"] < 0:
            raise ProjectIntegrityError("candidate-set artifact byte_size must be non-negative")
        set_records.append(record)

    ledger_records: list[dict[str, Any]] = []
    seen_ledger_ids: set[str] = set()
    for item in _sequence(value["review_ledgers"], "review_ledgers"):
        record = _object(
            item,
            "ReviewLedgerIndexEntry",
            {
                "review_ledger_id",
                "schema_version",
                "relative_path",
                "sha256",
                "byte_size",
            },
        )
        if type(record["review_ledger_id"]) is not str or _RUN_ID_RE.fullmatch(
            record["review_ledger_id"]
        ) is None:
            raise ProjectIntegrityError("review_ledger_id must be a lower-case opaque identifier")
        if record["review_ledger_id"] in seen_ledger_ids:
            raise ProjectIntegrityError("candidate store index has duplicate review-ledger ids")
        seen_ledger_ids.add(record["review_ledger_id"])
        if record["schema_version"] != TOWER_CANDIDATE_REVIEW_LEDGER_SCHEMA_VERSION:
            raise ProjectIntegrityError("unsupported review-ledger artifact schema version")
        _validate_candidate_store_relative_path(
            record["relative_path"], "candidates/review-ledgers/", "review-ledger relative_path"
        )
        _validate_digest(record["sha256"], "review-ledger artifact sha256")
        if type(record["byte_size"]) is not int or record["byte_size"] < 0:
            raise ProjectIntegrityError("review-ledger artifact byte_size must be non-negative")
        ledger_records.append(record)

    confirmation_records: list[dict[str, Any]] = []
    for item in _sequence(value["confirmations"], "confirmations"):
        record = _object(
            item,
            "ConfirmationIndexEntry",
            {
                "candidate_set_id",
                "candidate_id",
                "accepted_review_revision",
                "confirmed_tower_id",
                "confirmed_tower_revision",
                "schema_version",
                "relative_path",
                "sha256",
                "byte_size",
            },
        )
        if type(record["candidate_set_id"]) is not str or _RUN_ID_RE.fullmatch(
            record["candidate_set_id"]
        ) is None:
            raise ProjectIntegrityError("candidate_set_id must be a lower-case opaque identifier")
        if type(record["candidate_id"]) is not str or _CANDIDATE_ID_RE.fullmatch(
            record["candidate_id"]
        ) is None:
            raise ProjectIntegrityError("confirmation candidate_id must be a canonical P###")
        if type(record["accepted_review_revision"]) is not int or record[
            "accepted_review_revision"
        ] <= 0:
            raise ProjectIntegrityError("accepted_review_revision must be a positive integer")
        if type(record["confirmed_tower_id"]) is not str or _TOWER_ID_RE.fullmatch(
            record["confirmed_tower_id"]
        ) is None:
            raise ProjectIntegrityError("confirmed_tower_id must be a canonical tower ID")
        if type(record["confirmed_tower_revision"]) is not int or record[
            "confirmed_tower_revision"
        ] <= 0:
            raise ProjectIntegrityError("confirmed_tower_revision must be a positive integer")
        if record["schema_version"] != TOWER_CANDIDATE_CONFIRMATION_SCHEMA_VERSION:
            raise ProjectIntegrityError("unsupported confirmation artifact schema version")
        _validate_candidate_store_relative_path(
            record["relative_path"], "candidates/confirmations/", "confirmation relative_path"
        )
        _validate_digest(record["sha256"], "confirmation artifact sha256")
        if type(record["byte_size"]) is not int or record["byte_size"] < 0:
            raise ProjectIntegrityError("confirmation artifact byte_size must be non-negative")
        confirmation_records.append(record)

    return {
        "schema_version": value["schema_version"],
        "candidate_sets": tuple(set_records),
        "review_ledgers": tuple(ledger_records),
        "confirmations": tuple(confirmation_records),
    }


def _read_generation_artifact(
    generation_path: Path, record: Mapping[str, Any], name: str
) -> bytes:
    path = _confined_path(generation_path, record["relative_path"], name)
    try:
        data = path.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise FileIntegrityError(f"missing or unreadable {name} {record['relative_path']}") from exc
    if file_sha256(data) != record["sha256"]:
        raise FileIntegrityError(f"wrong SHA-256 for {name} {record['relative_path']}")
    if len(data) != record["byte_size"]:
        raise FileIntegrityError(f"wrong byte size for {name} {record['relative_path']}")
    return data


def _read_candidate_store(
    generation_path: Path,
    project: ProjectContract,
    generation_id: str,
    *,
    manifest_v2: bool,
) -> CandidateStore:
    candidates_dir = generation_path / "candidates"
    if not manifest_v2:
        if candidates_dir.exists():
            raise ProjectIntegrityError(
                "a legacy v1 generation must not contain a candidate store"
            )
        return CandidateStore()
    index_path = candidates_dir / "index.json"
    if not candidates_dir.exists() or not index_path.exists():
        raise ProjectIntegrityError("a v2 generation requires candidates/index.json")
    index = _decode_candidate_store_index(
        _read_json(index_path, "candidates/index.json"), project, generation_id
    )
    sets: list[TowerCandidateSet] = []
    for record in index["candidate_sets"]:
        expected = _candidate_set_relative_path(record["candidate_set_id"])
        if record["relative_path"] != expected:
            raise ProjectIntegrityError("candidate-set index relative_path is not canonical")
        data = _read_generation_artifact(generation_path, record, "candidate-set artifact")
        try:
            candidate_set = TowerCandidateSet.from_dict(
                _parse_json_bytes(data, record["relative_path"])
            )
        except (TypeError, ValueError) as exc:
            raise ProjectIntegrityError(f"invalid tower candidate-set artifact: {exc}") from exc
        if (
            candidate_set.candidate_set_id != record["candidate_set_id"]
            or candidate_set.digest != record["candidate_set_digest"]
        ):
            raise ProjectIntegrityError(
                "candidate-set artifact identity does not match the candidate index"
            )
        sets.append(candidate_set)
    ledgers: list[TowerCandidateReviewLedger] = []
    for record in index["review_ledgers"]:
        expected = _review_ledger_relative_path(record["review_ledger_id"])
        if record["relative_path"] != expected:
            raise ProjectIntegrityError("review-ledger index relative_path is not canonical")
        data = _read_generation_artifact(generation_path, record, "review-ledger artifact")
        try:
            ledger = TowerCandidateReviewLedger.from_dict(
                _parse_json_bytes(data, record["relative_path"])
            )
        except (TypeError, ValueError) as exc:
            raise ProjectIntegrityError(
                f"invalid tower candidate review-ledger artifact: {exc}"
            ) from exc
        if ledger.review_ledger_id != record["review_ledger_id"]:
            raise ProjectIntegrityError(
                "review-ledger artifact identity does not match the candidate index"
            )
        ledgers.append(ledger)
    confirmations: list[TowerCandidateConfirmation] = []
    for record in index["confirmations"]:
        expected = _confirmation_relative_path(
            record["candidate_set_id"],
            record["candidate_id"],
            record["accepted_review_revision"],
        )
        if record["relative_path"] != expected:
            raise ProjectIntegrityError("confirmation index relative_path is not canonical")
        data = _read_generation_artifact(generation_path, record, "confirmation artifact")
        try:
            confirmation = TowerCandidateConfirmation.from_dict(
                _parse_json_bytes(data, record["relative_path"])
            )
        except (TypeError, ValueError) as exc:
            raise ProjectIntegrityError(
                f"invalid tower candidate confirmation artifact: {exc}"
            ) from exc
        if (
            confirmation.candidate_set_reference.candidate_set_id != record["candidate_set_id"]
            or confirmation.candidate_id != record["candidate_id"]
            or confirmation.accepted_review_revision != record["accepted_review_revision"]
            or confirmation.confirmed_tower.tower_id != record["confirmed_tower_id"]
            or confirmation.confirmed_tower.revision != record["confirmed_tower_revision"]
        ):
            raise ProjectIntegrityError(
                "confirmation artifact identity does not match the candidate index"
            )
        confirmations.append(confirmation)
    try:
        store = CandidateStore(
            schema_version=index["schema_version"],
            candidate_sets=tuple(sets),
            review_ledgers=tuple(ledgers),
            confirmations=tuple(confirmations),
        )
    except PersistenceError as exc:
        raise ProjectIntegrityError(f"invalid candidate store: {exc}") from exc
    return store


def _validate_candidate_store_manifest_coverage(
    manifest: GenerationManifest,
    candidate_store: CandidateStore,
) -> None:
    if manifest.schema_version == GENERATION_SCHEMA_VERSION:
        # A historical v1 generation has no candidate-store files at all.
        expected: tuple[str, ...] = ()
    else:
        expected = _candidate_store_relative_paths(candidate_store)
    actual = tuple(
        item.relative_path
        for item in manifest.files
        if item.relative_path.startswith("candidates/")
    )
    if actual != expected:
        raise ProjectIntegrityError(
            "generation manifest candidate store files do not match the candidate store index"
        )


def _validate_candidate_store_state(
    project: ProjectContract,
    source_sidecar: SourceReferenceSidecar,
    tower_registry: TowerRegistry,
    candidate_store: CandidateStore,
) -> None:
    """Validate candidate artifacts against project/source identity and the Tower registry."""
    current_binding = project.source_reference.binding
    allowed_record_digests = {
        current_binding.source_identity_record_digest,
        *(item.source_identity_record_digest for item in source_sidecar.source_identity_history),
    }
    expected_coordinate_contexts = {
        current_binding.source_identity_record_digest: current_binding.coordinate_context
    }
    for history_identity in source_sidecar.source_identity_history:
        expected_coordinate_contexts[history_identity.source_identity_record_digest] = (
            SourceIdentityBinding.from_source_identity(history_identity).coordinate_context
        )
    for candidate_set in candidate_store.candidate_sets:
        if candidate_set.project_id != project.project_id:
            raise ProjectIntegrityError("candidate set project_id does not match project")
        if (
            candidate_set.source_binding.source_content_identity_digest
            != current_binding.source_content_identity_digest
        ):
            raise ProjectIntegrityError(
                "candidate set source content identity does not match project source"
            )
        if candidate_set.source_binding.source_identity_record_digest not in allowed_record_digests:
            raise ProjectIntegrityError(
                "candidate set source identity record is not in project source history"
            )
        if (
            candidate_set.coordinate_context
            != expected_coordinate_contexts[
                candidate_set.source_binding.source_identity_record_digest
            ]
        ):
            raise ProjectIntegrityError(
                "candidate set coordinate context does not match its referenced source identity"
            )
    for confirmation in candidate_store.confirmations:
        reference = confirmation.candidate_set_reference
        actual_set = next(
            (
                item
                for item in candidate_store.candidate_sets
                if item.candidate_set_id == reference.candidate_set_id
            ),
            None,
        )
        if actual_set is None or actual_set.reference != reference:
            raise ProjectIntegrityError(
                "confirmation candidate-set reference does not resolve to the actual set"
            )
        try:
            entry = tower_registry.get(confirmation.confirmed_tower.tower_id)
            revision = entry.revision(confirmation.confirmed_tower.revision)
        except KeyError as exc:
            raise ProjectIntegrityError(
                "confirmation references an unknown tower or tower revision"
            ) from exc
        if revision.source_binding != actual_set.source_binding:
            raise ProjectIntegrityError("confirmation tower source binding does not agree")
        evidence = revision.refinement.evidence
        if not (
            evidence.get("candidate_set_id")
            == confirmation.candidate_set_reference.candidate_set_id
            and evidence.get("candidate_set_digest")
            == confirmation.candidate_set_reference.candidate_set_digest
            and evidence.get("candidate_id") == confirmation.candidate_id
            and evidence.get("accepted_review_revision")
            == confirmation.accepted_review_revision
        ):
            raise ProjectIntegrityError(
                "confirmation tower provenance does not agree with the candidate review"
            )


def _decode_manifest(value: Any, project_id: str, generation_id: str) -> GenerationManifest:
    data = _object(
        value,
        "GenerationManifest",
        {
            "schema_version",
            "project_id",
            "generation_id",
            "sealed",
            "files",
            "project_manifest_logical_digest",
            "created_at",
            "published_at",
        },
    )
    manifest_version = data["schema_version"]
    if manifest_version not in {GENERATION_SCHEMA_VERSION, GENERATION_SCHEMA_VERSION_V2}:
        raise ProjectIntegrityError("unsupported generation schema version")
    if data["project_id"] != project_id:
        raise ProjectIntegrityError("generation manifest project_id does not match project")
    if data["generation_id"] != generation_id:
        raise ProjectIntegrityError("generation manifest generation_id does not match directory")
    if _bool(data["sealed"], "sealed") is not True:
        raise ProjectIntegrityError("generation manifest must be sealed")
    records: list[FileRecord] = []
    for item in _sequence(data["files"], "files"):
        record = _object(item, "FileRecord", {"relative_path", "byte_size", "sha256"})
        if type(record["relative_path"]) is not str:
            raise ProjectIntegrityError("file record relative_path must be a string")
        if type(record["byte_size"]) is not int or record["byte_size"] < 0:
            raise ProjectIntegrityError("file record byte_size must be a non-negative integer")
        records.append(
            FileRecord(
                relative_path=record["relative_path"],
                byte_size=record["byte_size"],
                sha256=_validate_digest(record["sha256"], "file record sha256"),
            )
        )
    if manifest_version == GENERATION_SCHEMA_VERSION:
        # Historical v1 manifests are the exact five core project files only.
        if tuple(item.relative_path for item in records) != _GENERATION_FILES:
            raise ProjectIntegrityError(
                "generation manifest v1 file records are not the exact historical five files"
            )
    else:
        # v2 preserves the five core files and adds the candidate review store.
        if (
            tuple(item.relative_path for item in records[: len(_GENERATION_FILES)])
            != _GENERATION_FILES
        ):
            raise ProjectIntegrityError(
                "generation manifest v2 core file records are not exact and ordered"
            )
        extra_files = [item.relative_path for item in records[len(_GENERATION_FILES) :]]
        if not extra_files or extra_files[0] != "candidates/index.json":
            raise ProjectIntegrityError(
                "generation manifest v2 must begin with candidates/index.json"
            )
        if any(not path.startswith("candidates/") for path in extra_files):
            raise ProjectIntegrityError(
                "generation manifest v2 references a file outside core and candidate store"
            )
    _validate_digest(data["project_manifest_logical_digest"], "project_manifest_logical_digest")
    _timestamp(data["created_at"], "created_at")
    _timestamp(data["published_at"], "published_at")
    return GenerationManifest(
        schema_version=data["schema_version"],
        project_id=data["project_id"],
        generation_id=data["generation_id"],
        sealed=True,
        files=tuple(records),
        project_manifest_logical_digest=data["project_manifest_logical_digest"],
        created_at=data["created_at"],
        published_at=data["published_at"],
    )


def _validate_generation_files(
    generation_path: Path,
    manifest: GenerationManifest,
) -> None:
    for record in manifest.files:
        path = _confined_path(generation_path, record.relative_path, "generation file reference")
        try:
            data = path.read_bytes()
        except (FileNotFoundError, OSError) as exc:
            raise FileIntegrityError(f"missing generation file {record.relative_path}") from exc
        if len(data) != record.byte_size:
            raise FileIntegrityError(f"wrong byte size for {record.relative_path}")
        if file_sha256(data) != record.sha256:
            raise FileIntegrityError(f"wrong SHA-256 for {record.relative_path}")


def _id_sequence(value: str, pattern: re.Pattern[str], name: str) -> int:
    match = pattern.fullmatch(value)
    if match is None:
        raise ProjectIntegrityError(f"{name} is not a canonical project display ID")
    sequence = int(match.group(1))
    if sequence <= 0:
        raise ProjectIntegrityError(f"{name} sequence must be positive")
    expected = (
        tower_display_id_for_sequence(sequence)
        if pattern is _TOWER_ID_RE
        else span_display_id_for_sequence(sequence)
    )
    if value != expected:
        raise ProjectIntegrityError(f"{name} is not the approved display ID for its sequence")
    return sequence


def _validate_registry_state(
    project: ProjectContract,
    source_sidecar: SourceReferenceSidecar,
    tower_registry: TowerRegistry,
    span_registry: SpanRegistry,
) -> None:
    """Validate counters, provenance, history, and all Tower/Span references."""

    current_binding = project.source_reference.binding
    allowed_record_digests = {
        current_binding.source_identity_record_digest,
        *(item.source_identity_record_digest for item in source_sidecar.source_identity_history),
    }
    # A historical Tower/Span binding legitimately references an earlier audited
    # source access record after an exact-content relocation.  Each binding's
    # coordinate context must match the source identity it actually references
    # (the current record or a validated record in source_identity_history),
    # never a different access record.
    expected_coordinate_contexts = {
        current_binding.source_identity_record_digest: current_binding.coordinate_context
    }
    for history_identity in source_sidecar.source_identity_history:
        expected_coordinate_contexts[history_identity.source_identity_record_digest] = (
            SourceIdentityBinding.from_source_identity(history_identity).coordinate_context
        )

    tower_sequences: set[int] = set()
    tower_by_id: dict[str, TowerRegistryEntry] = {}
    for entry in tower_registry.records:
        sequence = _id_sequence(entry.tower_id, _TOWER_ID_RE, "tower_id")
        tower_sequences.add(sequence)
        tower_by_id[entry.tower_id] = entry
        archived_seen = False
        for revision in entry.revisions:
            if revision.source_binding.source_content_identity_digest != current_binding.source_content_identity_digest:
                raise ProjectIntegrityError("tower source content identity does not match project source")
            if revision.source_binding.source_identity_record_digest not in allowed_record_digests:
                raise ProjectIntegrityError("tower source identity record is not in project source history")
            if (
                revision.coordinate_context
                != expected_coordinate_contexts[revision.source_binding.source_identity_record_digest]
            ):
                raise ProjectIntegrityError(
                    "tower coordinate context does not match its referenced source identity"
                )
            if revision.state is TowerLifecycleState.ARCHIVED:
                archived_seen = True
            elif archived_seen:
                raise ProjectIntegrityError("an archived tower cannot receive a later confirmed revision")

    tower_reserved = {
        _id_sequence(item["display_id"], _TOWER_ID_RE, "tower reservation display_id")
        for item in tower_registry.reserved_sequences
    }
    tower_sequences.update(tower_reserved)
    if project.next_tower_sequence <= max(tower_sequences, default=0):
        raise ProjectIntegrityError("next_tower_sequence is not above every represented/reserved tower")

    span_sequences: set[int] = set()
    seen_active_physical_pairs: set[frozenset[str]] = set()
    span_ids: set[str] = set()
    for span in span_registry.records:
        span_sequence = _id_sequence(span.span_id, _SPAN_ID_RE, "span_id")
        span_sequences.add(span_sequence)
        if span.span_id in span_ids:
            raise ProjectIntegrityError("span registry contains duplicate span_id")
        span_ids.add(span.span_id)
        if span.source_binding.source_content_identity_digest != current_binding.source_content_identity_digest:
            raise ProjectIntegrityError("span source content identity does not match project source")
        if span.source_binding.source_identity_record_digest not in allowed_record_digests:
            raise ProjectIntegrityError("span source identity record is not in project source history")
        if (
            span.source_binding.coordinate_context
            != expected_coordinate_contexts[span.source_binding.source_identity_record_digest]
        ):
            raise ProjectIntegrityError(
                "span coordinate context does not match its referenced source identity"
            )
        from_entry = tower_by_id.get(span.from_tower_id)
        to_entry = tower_by_id.get(span.to_tower_id)
        if from_entry is None or to_entry is None:
            raise ProjectIntegrityError("span endpoint tower does not exist")
        try:
            from_entry.revision(span.from_tower_revision)
            to_entry.revision(span.to_tower_revision)
        except KeyError as exc:
            raise ProjectIntegrityError("span endpoint revision does not exist") from exc
        if span.topology_state is SpanTopologyState.CONFIRMED:
            if (
                from_entry.current.state is TowerLifecycleState.ARCHIVED
                or to_entry.current.state is TowerLifecycleState.ARCHIVED
            ):
                raise ProjectIntegrityError("active span cannot reference an archived tower")
            pair = frozenset((span.from_tower_id, span.to_tower_id))
            if pair in seen_active_physical_pairs:
                raise ProjectIntegrityError("duplicate active physical span topology")
            seen_active_physical_pairs.add(pair)

    span_reserved = {
        _id_sequence(item["display_id"], _SPAN_ID_RE, "span reservation display_id")
        for item in span_registry.reserved_sequences
    }
    span_sequences.update(span_reserved)
    if project.next_span_sequence <= max(span_sequences, default=0):
        raise ProjectIntegrityError("next_span_sequence is not above every represented/reserved span")


def _validate_run_index_state(
    project: ProjectContract,
    source_sidecar: SourceReferenceSidecar,
    tower_registry: TowerRegistry,
    span_registry: SpanRegistry,
    run_index: RunIndex,
) -> None:
    """Validate historical run provenance without rewriting historical snapshots."""

    current_binding = project.source_reference.binding
    allowed_record_digests = {
        current_binding.source_identity_record_digest,
        *(item.source_identity_record_digest for item in source_sidecar.source_identity_history),
    }
    expected_coordinate_contexts = {
        current_binding.source_identity_record_digest: current_binding.coordinate_context
    }
    for history_identity in source_sidecar.source_identity_history:
        expected_coordinate_contexts[history_identity.source_identity_record_digest] = (
            SourceIdentityBinding.from_source_identity(history_identity).coordinate_context
        )
    span_by_id = {span.span_id: span for span in span_registry.records}
    tower_by_id = {entry.tower_id: entry for entry in tower_registry.records}

    for entry in run_index.entries:
        span = span_by_id.get(entry.span_id)
        if span is None:
            raise ProjectIntegrityError("run index entry references an unknown span")
        dependencies = entry.span_dependencies
        binding = dependencies.source_binding
        if entry.source_content_identity_digest != binding.source_content_identity_digest:
            raise ProjectIntegrityError(
                "run source_content_identity_digest does not match dependency snapshot"
            )
        if entry.source_identity_record_digest != binding.source_identity_record_digest:
            raise ProjectIntegrityError(
                "run source_identity_record_digest does not match dependency snapshot"
            )
        if binding.source_content_identity_digest != current_binding.source_content_identity_digest:
            raise ProjectIntegrityError("run source content identity does not match project source")
        if binding.source_identity_record_digest not in allowed_record_digests:
            raise ProjectIntegrityError("run source identity record is not in project source history")
        if (
            binding.coordinate_context
            != expected_coordinate_contexts[binding.source_identity_record_digest]
        ):
            raise ProjectIntegrityError(
                "run coordinate context does not match its referenced source identity"
            )
        try:
            tower_by_id[dependencies.from_tower.tower_id].revision(
                dependencies.from_tower.revision
            )
            tower_by_id[dependencies.to_tower.tower_id].revision(
                dependencies.to_tower.revision
            )
        except KeyError as exc:
            raise ProjectIntegrityError(
                "run dependency references an unknown tower or tower revision"
            ) from exc
        if (
            entry.status is AnalysisStatus.STALE
            and dependencies == span.current_dependencies
        ):
            raise ProjectIntegrityError(
                "STALE run dependencies must differ from the current associated span"
            )


def _validate_run_index_span_analysis_coherence(
    span_registry: SpanRegistry,
    run_index: RunIndex,
) -> None:
    """Fail closed unless every Span is the exact projection of its current run.

    Run-index records are historical evidence.  ``SpanAnalysis`` is instead the
    current/latest per-Span projection: it names one active run, mirrors that
    run's status and immutable dependencies, and lists prior runs in
    ``historical_run_ids``. Terminal active runs are additionally retained in
    that history; P1 independently requires this for STALE.
    """

    entries_by_id = {entry.run_id: entry for entry in run_index.entries}
    run_ids_by_span: dict[str, set[str]] = {}
    for entry in run_index.entries:
        run_ids_by_span.setdefault(entry.span_id, set()).add(entry.run_id)

    for span in span_registry.records:
        analysis = span.analysis
        run_ids = run_ids_by_span.get(span.span_id, set())
        historical_ids = set(analysis.historical_run_ids)
        active_run_id = analysis.active_run_id
        if active_run_id is None:
            if run_ids:
                raise ProjectIntegrityError(
                    "Span with run-index entries must retain its current active_run_id"
                )
            if historical_ids:
                raise ProjectIntegrityError(
                    "Span without run-index entries must not retain historical_run_ids"
                )
            # P1 permits a no-run READY preparation state as well as the normal
            # NOT_STARTED state.  Both have no run provenance and are coherent.
            continue

        active_entry = entries_by_id.get(active_run_id)
        if active_entry is None:
            raise ProjectIntegrityError("Span active_run_id does not resolve in the run index")
        if active_entry.span_id != span.span_id:
            raise ProjectIntegrityError("Span active_run_id belongs to a different Span")
        if analysis.state is not active_entry.status:
            raise ProjectIntegrityError("Span analysis status does not match its active run")
        if analysis.run_dependencies != active_entry.span_dependencies:
            raise ProjectIntegrityError(
                "Span analysis dependencies do not match its active run snapshot"
            )

        expected_history = set(run_ids)
        expected_history.discard(active_run_id)
        if active_entry.status in {
            AnalysisStatus.COMPLETE,
            AnalysisStatus.FAILED,
            AnalysisStatus.INTERRUPTED,
            AnalysisStatus.STALE,
        }:
            expected_history.add(active_run_id)
        if historical_ids != expected_history:
            raise ProjectIntegrityError(
                "Span historical_run_ids do not match its run-index history"
            )


def _validate_generation_content(
    root: Path,
    generation_path: Path,
    generation_id: str,
    project: ProjectContract,
    source_sidecar: SourceReferenceSidecar,
    tower_registry: TowerRegistry,
    span_registry: SpanRegistry,
    run_index: RunIndex,
    candidate_store: CandidateStore,
) -> None:
    if source_sidecar.project_id != project.project_id:
        raise ProjectIntegrityError("source sidecar project_id does not match project")
    if source_sidecar.generation_id != generation_id:
        raise ProjectIntegrityError("source sidecar generation_id does not match generation")
    if source_sidecar.project_source_reference != project.source_reference:
        raise ProjectIntegrityError("source sidecar source reference does not match project")
    for field_name in _PROJECT_REGISTRY_REFERENCES:
        _project_reference_path(project, field_name, root, generation_path, generation_id)
    if (
        tower_registry.project_id != project.project_id
        or span_registry.project_id != project.project_id
    ):
        raise ProjectIntegrityError("registry project identity does not match project")
    if run_index.project_id != project.project_id:
        raise ProjectIntegrityError("run index project identity does not match project")
    _validate_registry_state(project, source_sidecar, tower_registry, span_registry)
    _validate_run_index_state(
        project, source_sidecar, tower_registry, span_registry, run_index
    )
    _validate_run_index_span_analysis_coherence(span_registry, run_index)
    _validate_candidate_store_state(project, source_sidecar, tower_registry, candidate_store)


def _read_generation_content(
    root: Path,
    generation_path: Path,
    generation_id: str,
    *,
    manifest_v2: bool,
) -> tuple[
    ProjectContract,
    SourceReferenceSidecar,
    TowerRegistry,
    SpanRegistry,
    RunIndex,
    CandidateStore,
]:
    project_data = _read_json(generation_path / "project.json", "generation project.json")
    project = _decode_project(project_data)
    source_data = _read_json(
        generation_path / "source" / "source-reference.json",
        "source/source-reference.json",
    )
    source_sidecar = _decode_source_sidecar(source_data)
    tower_data = _read_json(generation_path / "towers" / "towers.json", "towers/towers.json")
    tower_registry = _decode_registry(tower_data, project, generation_id)
    if not isinstance(tower_registry, TowerRegistry):
        raise ProjectIntegrityError("towers/towers.json has the wrong registry schema")
    span_data = _read_json(generation_path / "spans" / "spans.json", "spans/spans.json")
    span_registry = _decode_registry(span_data, project, generation_id)
    if not isinstance(span_registry, SpanRegistry):
        raise ProjectIntegrityError("spans/spans.json has the wrong registry schema")
    run_data = _read_json(generation_path / "runs" / "run-index.json", "runs/run-index.json")
    run_index = _decode_run_index(run_data, project, generation_id)
    candidate_store = _read_candidate_store(
        generation_path, project, generation_id, manifest_v2=manifest_v2
    )
    _validate_generation_content(
        root,
        generation_path,
        generation_id,
        project,
        source_sidecar,
        tower_registry,
        span_registry,
        run_index,
        candidate_store,
    )
    return project, source_sidecar, tower_registry, span_registry, run_index, candidate_store


def _read_root_pointer(root: Path) -> tuple[dict[str, Any], ProjectContract]:
    pointer_data = _read_json(root / "project.json", "root project.json")
    pointer = _object(
        pointer_data,
        "ActivePointer",
        {
            "schema_version",
            "project_id",
            "active_generation_id",
            "generation_manifest_reference",
            "generation_manifest_sha256",
            "project_manifest_reference",
            "project_manifest_sha256",
            "project",
        },
    )
    if pointer["schema_version"] != ACTIVE_POINTER_SCHEMA_VERSION:
        raise ProjectIntegrityError("unsupported active-pointer schema version")
    generation_id = _validate_generation_id(pointer["active_generation_id"])
    _validate_digest(pointer["generation_manifest_sha256"], "generation_manifest_sha256")
    _validate_digest(pointer["project_manifest_sha256"], "project_manifest_sha256")
    project = _decode_project(pointer["project"])
    if pointer["project_id"] != project.project_id:
        raise ProjectIntegrityError("active pointer project_id does not match nested project")
    expected_generation_reference = f"state/generations/{generation_id}/generation.json"
    expected_project_reference = f"state/generations/{generation_id}/project.json"
    if pointer["generation_manifest_reference"] != expected_generation_reference:
        raise ProjectIntegrityError("active pointer generation reference is unsafe or mismatched")
    if pointer["project_manifest_reference"] != expected_project_reference:
        raise ProjectIntegrityError("active pointer project reference is unsafe or mismatched")
    _confined_path(root, pointer["generation_manifest_reference"], "generation reference")
    _confined_path(root, pointer["project_manifest_reference"], "project reference")
    return pointer, project


def open_project(project_root: str | os.PathLike[str]) -> LoadedProject:
    """Strictly reopen exactly the generation selected by the root pointer."""

    root = Path(project_root)
    if not root.is_dir():
        raise ProjectIntegrityError("project root must be an existing directory")
    pointer, pointer_project = _read_root_pointer(root)
    generation_id = pointer["active_generation_id"]
    generation_path = _generation_path(root, generation_id)
    if not generation_path.is_dir() or generation_path.name.startswith(_STAGING_PREFIX):
        raise ProjectIntegrityError("active generation is missing or is a staging directory")

    manifest_path = generation_path / "generation.json"
    manifest_data = _read_json(manifest_path, "generation.json")
    manifest_bytes = manifest_path.read_bytes()
    if file_sha256(manifest_bytes) != pointer["generation_manifest_sha256"]:
        raise FileIntegrityError("root pointer generation manifest digest mismatch")
    manifest = _decode_manifest(manifest_data, pointer_project.project_id, generation_id)
    _validate_generation_files(generation_path, manifest)
    project_path = generation_path / "project.json"
    project_bytes = project_path.read_bytes()
    if file_sha256(project_bytes) != pointer["project_manifest_sha256"]:
        raise FileIntegrityError("root pointer project manifest digest mismatch")
    project, source_sidecar, tower_registry, span_registry, run_index, candidate_store = (
        _read_generation_content(
            root,
            generation_path,
            generation_id,
            manifest_v2=manifest.schema_version == GENERATION_SCHEMA_VERSION_V2,
        )
    )
    _validate_candidate_store_manifest_coverage(manifest, candidate_store)
    if project != pointer_project:
        raise ProjectIntegrityError("root pointer nested project does not match generation project")
    if project.project_id != manifest.project_id or project.project_id != pointer["project_id"]:
        raise ProjectIntegrityError("project identity is inconsistent across the active generation")
    project_payload = project.to_dict()
    if manifest.project_manifest_logical_digest != logical_json_digest(project_payload):
        raise FileIntegrityError("wrong logical project manifest digest")
    if canonical_json_bytes(project_payload) + b"\n" != project_bytes:
        raise FileIntegrityError("generation project manifest changed after validation")
    return LoadedProject(
        project=project,
        source_sidecar=source_sidecar,
        tower_registry=tower_registry,
        span_registry=span_registry,
        run_index=run_index,
        candidate_store=candidate_store,
        generation_id=generation_id,
        generation_manifest=manifest,
        project_root=root,
        generation_path=generation_path,
    )


def _write_candidate_store_files(
    staging_path: Path,
    project_id: str,
    generation_id: str,
    candidate_store: CandidateStore,
) -> None:
    """Write immutable candidate artifacts plus their digest-protected index.

    Every v2 generation carries ``candidates/index.json`` even when the store
    is currently empty, so an empty v2 generation is unambiguous.
    """
    candidates_dir = staging_path / "candidates"
    candidates_dir.mkdir()
    (candidates_dir / "candidate-sets").mkdir()
    (candidates_dir / "review-ledgers").mkdir()
    (candidates_dir / "confirmations").mkdir()
    set_records: list[dict[str, Any]] = []
    for item in sorted(candidate_store.candidate_sets, key=lambda s: s.candidate_set_id):
        relative = _candidate_set_relative_path(item.candidate_set_id)
        data = _write_json(staging_path / relative, item.to_dict())
        set_records.append(
            {
                "candidate_set_id": item.candidate_set_id,
                "candidate_set_digest": item.digest,
                "schema_version": item.schema_version,
                "relative_path": relative,
                "sha256": file_sha256(data),
                "byte_size": len(data),
            }
        )
    ledger_records: list[dict[str, Any]] = []
    for item in sorted(candidate_store.review_ledgers, key=lambda ledger: ledger.review_ledger_id):
        relative = _review_ledger_relative_path(item.review_ledger_id)
        data = _write_json(staging_path / relative, item.to_dict())
        ledger_records.append(
            {
                "review_ledger_id": item.review_ledger_id,
                "schema_version": item.schema_version,
                "relative_path": relative,
                "sha256": file_sha256(data),
                "byte_size": len(data),
            }
        )
    confirmation_records: list[dict[str, Any]] = []
    for item in sorted(
        candidate_store.confirmations,
        key=lambda confirmation: (
            confirmation.candidate_set_reference.candidate_set_id,
            confirmation.candidate_id,
            confirmation.accepted_review_revision,
        ),
    ):
        reference = item.candidate_set_reference
        relative = _confirmation_relative_path(
            reference.candidate_set_id, item.candidate_id, item.accepted_review_revision
        )
        data = _write_json(staging_path / relative, item.to_dict())
        confirmation_records.append(
            {
                "candidate_set_id": reference.candidate_set_id,
                "candidate_id": item.candidate_id,
                "accepted_review_revision": item.accepted_review_revision,
                "confirmed_tower_id": item.confirmed_tower.tower_id,
                "confirmed_tower_revision": item.confirmed_tower.revision,
                "schema_version": item.schema_version,
                "relative_path": relative,
                "sha256": file_sha256(data),
                "byte_size": len(data),
            }
        )
    _write_json(
        candidates_dir / "index.json",
        {
            "schema_version": CANDIDATE_STORE_INDEX_SCHEMA_VERSION,
            "project_id": project_id,
            "generation_id": generation_id,
            "candidate_sets": set_records,
            "review_ledgers": ledger_records,
            "confirmations": confirmation_records,
        },
    )
    for child_name in ("candidate-sets", "review-ledgers", "confirmations"):
        _fsync_directory(candidates_dir / child_name)
    _fsync_directory(candidates_dir)


def _write_generation_files(
    root: Path,
    staging_path: Path,
    generation_id: str,
    project: ProjectContract,
    source_sidecar: SourceReferenceSidecar,
    tower_registry: TowerRegistry | None = None,
    span_registry: SpanRegistry | None = None,
    run_index: RunIndex | None = None,
    candidate_store: CandidateStore | None = None,
) -> tuple[
    dict[str, Any],
    SourceReferenceSidecar,
    TowerRegistry,
    SpanRegistry,
    RunIndex,
    CandidateStore,
]:
    (staging_path / "source").mkdir()
    (staging_path / "towers").mkdir()
    (staging_path / "spans").mkdir()
    (staging_path / "runs").mkdir()
    tower_registry = TowerRegistry(
        TOWER_REGISTRY_SCHEMA_VERSION,
        generation_id,
        project.project_id,
        reserved_sequences=(tower_registry.reserved_sequences if tower_registry is not None else ()),
        records=(tower_registry.records if tower_registry is not None else ()),
    )
    span_registry = SpanRegistry(
        SPAN_REGISTRY_SCHEMA_VERSION,
        generation_id,
        project.project_id,
        reserved_sequences=(span_registry.reserved_sequences if span_registry is not None else ()),
        records=(span_registry.records if span_registry is not None else ()),
    )
    run_index = RunIndex(
        RUN_INDEX_SCHEMA_VERSION,
        generation_id,
        project.project_id,
        entries=(run_index.entries if run_index is not None else ()),
    )
    candidate_store = (
        CandidateStore()
        if candidate_store is None
        else CandidateStore(
            schema_version=candidate_store.schema_version,
            candidate_sets=candidate_store.candidate_sets,
            review_ledgers=candidate_store.review_ledgers,
            confirmations=candidate_store.confirmations,
        )
    )
    project_payload = project.to_dict()
    _write_json(staging_path / "source" / "source-reference.json", source_sidecar.to_dict())
    _write_json(staging_path / "towers" / "towers.json", tower_registry.to_dict())
    _write_json(staging_path / "spans" / "spans.json", span_registry.to_dict())
    _write_json(staging_path / "runs" / "run-index.json", run_index.to_dict())
    _write_json(staging_path / "project.json", project_payload)
    _write_candidate_store_files(staging_path, project.project_id, generation_id, candidate_store)
    for child_name in ("source", "towers", "spans", "runs"):
        _fsync_directory(staging_path / child_name)
    _fsync_directory(staging_path)
    # Staging is always published as generation v2.
    _read_generation_content(root, staging_path, generation_id, manifest_v2=True)
    return (
        project_payload,
        source_sidecar,
        tower_registry,
        span_registry,
        run_index,
        candidate_store,
    )


def _file_records(
    staging_path: Path, candidate_store: CandidateStore
) -> tuple[FileRecord, ...]:
    records: list[FileRecord] = []
    for relative in _GENERATION_FILES:
        path = _confined_path(staging_path, relative, "generation file")
        data = path.read_bytes()
        records.append(FileRecord(relative, len(data), file_sha256(data)))
    for relative in _candidate_store_relative_paths(candidate_store):
        path = _confined_path(staging_path, relative, "candidate store file")
        data = path.read_bytes()
        records.append(FileRecord(relative, len(data), file_sha256(data)))
    return tuple(records)


def _publish_generation(
    root: Path,
    project: ProjectContract,
    *,
    source_access_metadata: Mapping[str, Any] | None = None,
    source_sidecar: SourceReferenceSidecar | None = None,
    tower_registry: TowerRegistry | None = None,
    span_registry: SpanRegistry | None = None,
    run_index: RunIndex | None = None,
    candidate_store: CandidateStore | None = None,
) -> LoadedProject:
    """Copy-on-write publisher used by locked session mutation and P2A creation/tests."""

    if type(project) is not ProjectContract:
        raise PersistenceError("project must be an exact ProjectContract")
    # Exercise the strict boundary before any staging side effect.
    decoded_project = _decode_project(project.to_dict())
    if decoded_project != project:
        raise PersistenceError("project did not survive strict serialization round trip")
    root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise PersistenceError("project root must be a directory")
    generations = _generation_parent(root)
    generation_id = _next_generation_id(generations)
    staging_path = generations / f"{_STAGING_PREFIX}{uuid.uuid4().hex}"
    staging_path.mkdir()
    if source_sidecar is not None and source_access_metadata is not None:
        raise PersistenceError("source_sidecar and source_access_metadata are mutually exclusive")
    if source_sidecar is None:
        source_sidecar = _build_source_sidecar(project, generation_id, source_access_metadata)
    else:
        source_sidecar = _rebind_source_sidecar(
            source_sidecar, generation_id=generation_id, project=project
        )
    try:
        project_payload, _, _, _, _, candidate_store = _write_generation_files(
            root,
            staging_path,
            generation_id,
            project,
            source_sidecar,
            tower_registry,
            span_registry,
            run_index,
            candidate_store,
        )
        created_at = _utc_timestamp()
        manifest = GenerationManifest(
            schema_version=GENERATION_SCHEMA_VERSION_V2,
            project_id=project.project_id,
            generation_id=generation_id,
            sealed=True,
            files=_file_records(staging_path, candidate_store),
            project_manifest_logical_digest=logical_json_digest(project_payload),
            created_at=created_at,
            published_at=_utc_timestamp(),
        )
        _write_json(staging_path / "generation.json", manifest.to_dict())
        _fsync_directory(staging_path)
        # Re-read and validate the sealed staging generation before rename.
        staged_manifest = _decode_manifest(
            _read_json(staging_path / "generation.json", "staged generation.json"),
            project.project_id,
            generation_id,
        )
        _validate_generation_files(staging_path, staged_manifest)
        _read_generation_content(root, staging_path, generation_id, manifest_v2=True)
        final_path = generations / generation_id
        try:
            os.replace(staging_path, final_path)
        except OSError as exc:
            raise GenerationPublicationError(
                f"failed to publish generation directory: {generation_id}",
                operation="generation-directory-replace",
                path=final_path,
            ) from exc
        _fsync_directory(generations)
        # Validate the published immutable directory before changing the root pointer.
        _validate_published_generation(root, final_path, generation_id)
        pointer = {
            "schema_version": ACTIVE_POINTER_SCHEMA_VERSION,
            "project_id": project.project_id,
            "active_generation_id": generation_id,
            "generation_manifest_reference": f"state/generations/{generation_id}/generation.json",
            "generation_manifest_sha256": file_sha256(
                (final_path / "generation.json").read_bytes()
            ),
            "project_manifest_reference": f"state/generations/{generation_id}/project.json",
            "project_manifest_sha256": file_sha256((final_path / "project.json").read_bytes()),
            "project": project_payload,
        }
        _atomic_replace_root_pointer(root, pointer)
        return open_project(root)
    except Exception:
        # A failed stage is intentionally retained as an unreferenced staging
        # directory.  The old root pointer is never repaired or replaced here.
        raise


def _validate_published_generation(
    root: Path,
    generation_path: Path,
    generation_id: str,
) -> None:
    manifest_path = generation_path / "generation.json"
    project = _decode_project(
        _read_json(generation_path / "project.json", "generation project.json")
    )
    manifest = _decode_manifest(
        _read_json(manifest_path, "generation.json"),
        project.project_id,
        generation_id,
    )
    _validate_generation_files(generation_path, manifest)
    _, _, _, _, _, candidate_store = _read_generation_content(
        root,
        generation_path,
        generation_id,
        manifest_v2=manifest.schema_version == GENERATION_SCHEMA_VERSION_V2,
    )
    _validate_candidate_store_manifest_coverage(manifest, candidate_store)


def _atomic_replace_root_pointer(root: Path, pointer: Mapping[str, Any]) -> None:
    target = root / "project.json"
    temporary = root / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        try:
            _write_json(temporary, dict(pointer))
        except (DurabilityError, OSError) as exc:
            raise RootPointerReplaceError(
                "failed to durably write root active-pointer temporary file",
                operation="root-pointer-temp-write-fsync",
                path=temporary,
            ) from exc
        _read_json(temporary, "temporary root project.json")
        try:
            os.replace(temporary, target)
        except OSError as exc:
            raise RootPointerReplaceError(
                "failed to replace root active pointer; prior pointer remains authoritative",
                operation="root-pointer-replace",
                path=target,
            ) from exc
        try:
            _fsync_directory(root)
        except DirectoryDurabilityError as exc:
            raise PostPointerDurabilityError(
                "root pointer was replaced before project-root durability failed; "
                "reopen and revalidate before retrying",
                operation="root-directory-fsync-after-pointer-replace",
                path=root,
                pointer_replaced=True,
            ) from exc
        _read_json(target, "published root project.json")
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _create_project_unlocked_for_tests(
    project_root: str | os.PathLike[str],
    project: ProjectContract,
    *,
    source_access_metadata: Mapping[str, Any] | None = None,
) -> LoadedProject:
    """Create an initial immutable P2A generation for internal P2A tests.

    P2B owns the normal creation boundary through ``create_project_writable``.
    This helper remains private so P2A persistence tests can exercise its
    immutable-generation mechanics without becoming an unlocked application
    publication API.
    """

    root = Path(project_root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "project.json").exists():
        raise PersistenceError("project root already has an active project pointer")
    return _publish_generation(
        root,
        project,
        source_access_metadata=source_access_metadata,
    )


__all__ = [
    "ACTIVE_POINTER_SCHEMA_VERSION",
    "CANDIDATE_STORE_INDEX_SCHEMA_VERSION",
    "GENERATION_SCHEMA_VERSION",
    "GENERATION_SCHEMA_VERSION_V2",
    "RUN_INDEX_SCHEMA_VERSION",
    "SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION",
    "SPAN_REGISTRY_SCHEMA_VERSION",
    "TOWER_REGISTRY_SCHEMA_VERSION",
    "CandidateStore",
    "FileIntegrityError",
    "GenerationManifest",
    "LoadedProject",
    "PersistenceError",
    "ProjectIntegrityError",
    "PostPointerDurabilityError",
    "RunIndex",
    "RunIndexEntry",
    "SourceReferenceSidecar",
    "SpanRegistry",
    "TowerRegistry",
    "TowerRegistryEntry",
    "canonical_json_bytes",
    "deserialize_project",
    "file_sha256",
    "logical_json_digest",
    "open_project",
    "published_json_bytes",
    "published_json_digest",
    "project_from_dict",
]
