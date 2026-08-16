"""Authoritative P2B project-session lifecycle and source-availability gate.

The P2A loader remains the immutable generation validator.  This module adds
the boundary that decides *when* that loader may be used for a writable
authoritative session: only after the OS writer lock has been acquired.

Strong source verification here is an in-memory session gate.  It validates
the persisted ``SourceIdentity`` plus the source-file digest using bounded
streaming I/O; it neither materializes point arrays nor substitutes for the
final source-provider TOCTOU verification required by a future scientific
operation.
"""

from __future__ import annotations

import hashlib
import math
import os
import stat as stat_module
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from algorithms.io.headers import read_las_header_summary

from core.contracts.frv import (
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.project import ProjectContract
from core.project import persistence
from core.project.locking import (
    DEFAULT_APPLICATION_VERSION,
    ProjectWriterLock,
    acquire_project_writer_lock,
    prepare_project_root,
)
from core.project.persistence import LoadedProject, PersistenceError

_WRITABLE_SESSION_TOKEN = object()


class ProjectSessionError(RuntimeError):
    """Base error for P2B project-session lifecycle failures."""


class ProjectSessionClosedError(ProjectSessionError):
    """Raised when a closed session is used."""


class ProjectAlreadyExistsError(ProjectSessionError):
    """Raised when locked project creation targets an existing active project."""


class SourceAvailabilityError(ProjectSessionError):
    """Base error for a source-dependent operation that cannot proceed."""


class SourceMissingError(SourceAvailabilityError):
    """Raised when the persisted external source path no longer exists."""


class SourceIdentityMismatchError(SourceAvailabilityError):
    """Raised when a source cannot be the persisted source identity."""


class StrongSourceVerificationError(SourceAvailabilityError):
    """Raised when strong verification cannot complete safely."""


class ProjectSessionMode(str, Enum):
    WRITABLE = "WRITABLE"
    READ_ONLY = "READ_ONLY"


class SourceAvailability(str, Enum):
    SOURCE_MISSING = "SOURCE_MISSING"
    SOURCE_PRESENT_UNVERIFIED = "SOURCE_PRESENT_UNVERIFIED"
    SOURCE_IDENTITY_MISMATCH = "SOURCE_IDENTITY_MISMATCH"
    SOURCE_VERIFIED = "SOURCE_VERIFIED"


@dataclass(frozen=True, slots=True)
class SourceFileSignature:
    """Session-only source signature used to invalidate strong verification."""

    resolved_path: str
    expected_source_content_identity_digest: str
    file_size_bytes: int
    mtime_ns: int
    platform_file_identity: tuple[int, int] | None


@dataclass(frozen=True, slots=True)
class _StrongVerificationCache:
    signature: SourceFileSignature


def _source_path(loaded: LoadedProject) -> Path:
    raw_path = Path(loaded.project.source_reference.source_identity.resolved_access_path)
    if raw_path.is_absolute():
        return raw_path
    # P2A fixtures predate P2B's external-path rule and use a relative audited
    # path.  Interpret those legacy values only relative to this project root;
    # never search for similarly named files elsewhere.
    return loaded.project_root / raw_path


def _file_signature(loaded: LoadedProject) -> tuple[Path, SourceFileSignature]:
    path = _source_path(loaded)
    resolved = path.resolve(strict=True)
    file_stat = resolved.stat()
    if not stat_module.S_ISREG(file_stat.st_mode):
        raise OSError("source path is not a regular file")
    device = getattr(file_stat, "st_dev", None)
    inode = getattr(file_stat, "st_ino", None)
    identity = (
        (int(device), int(inode))
        if device is not None and inode is not None and int(inode) != 0
        else None
    )
    return resolved, SourceFileSignature(
        resolved_path=str(resolved),
        expected_source_content_identity_digest=(
            loaded.project.source_reference.source_identity.source_content_identity_digest
        ),
        file_size_bytes=int(file_stat.st_size),
        mtime_ns=int(file_stat.st_mtime_ns),
        platform_file_identity=identity,
    )


def _same_xyz(first: tuple[float, float, float], second: tuple[float, float, float]) -> bool:
    return all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)
        for left, right in zip(first, second, strict=True)
    )


def _fast_source_availability(
    loaded: LoadedProject,
) -> tuple[SourceAvailability, SourceFileSignature | None]:
    """Check path/stat/header invariants without hashing or point reads."""

    try:
        resolved_path, signature = _file_signature(loaded)
    except FileNotFoundError:
        return SourceAvailability.SOURCE_MISSING, None
    except OSError:
        return SourceAvailability.SOURCE_IDENTITY_MISMATCH, None

    if resolved_path.suffix.lower() not in {".las", ".laz"}:
        return SourceAvailability.SOURCE_IDENTITY_MISMATCH, signature

    identity = loaded.project.source_reference.source_identity
    access_metadata = loaded.source_sidecar.access_metadata
    if (
        access_metadata["source_available_at_save"]
        and signature.file_size_bytes != access_metadata["file_size_bytes"]
    ):
        return SourceAvailability.SOURCE_IDENTITY_MISMATCH, signature

    try:
        # This repository helper is intentionally header-only.  It uses
        # ``laspy.open(...).header`` and never materializes point arrays.
        header = read_las_header_summary(resolved_path, source_id=identity.source_id)
    except Exception:  # Header parsing failure proves this is not a safe compatible source.
        return SourceAvailability.SOURCE_IDENTITY_MISMATCH, signature
    if (
        header.point_count != identity.source_point_count
        or not _same_xyz(header.min_xyz, identity.source_bounds_min_xyz)
        or not _same_xyz(header.max_xyz, identity.source_bounds_max_xyz)
    ):
        return SourceAvailability.SOURCE_IDENTITY_MISMATCH, signature
    return SourceAvailability.SOURCE_PRESENT_UNVERIFIED, signature


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


class ProjectSession:
    """One immutable loaded P2A generation plus explicit P2B session state."""

    def __init__(
        self,
        loaded: LoadedProject,
        *,
        mode: ProjectSessionMode,
        writer_lock: ProjectWriterLock | None,
        application_version: str,
        _authority_token: object | None = None,
    ) -> None:
        if (mode is ProjectSessionMode.WRITABLE) != (writer_lock is not None):
            raise ValueError("writable sessions must own exactly one writer lock")
        if writer_lock is not None and type(writer_lock) is not ProjectWriterLock:
            raise ValueError("writer_lock must be an actual ProjectWriterLock")
        if mode is ProjectSessionMode.WRITABLE and _authority_token is not _WRITABLE_SESSION_TOKEN:
            raise ValueError(
                "writable sessions must be issued by the supported lock/open path"
            )
        if mode is ProjectSessionMode.READ_ONLY and _authority_token is not None:
            raise ValueError("read-only sessions cannot carry writable authority")
        if writer_lock is not None:
            writer_lock.assert_live_writer_authority(loaded.project_root)
        self._loaded = loaded
        self._mode = mode
        self._writer_lock = writer_lock
        self._application_version = application_version
        self._closed = False
        self._strong_cache: _StrongVerificationCache | None = None
        self._source_availability, self._last_source_signature = _fast_source_availability(loaded)

    def _require_open(self) -> None:
        if self._closed:
            raise ProjectSessionClosedError("project session is closed")

    @property
    def mode(self) -> ProjectSessionMode:
        self._require_open()
        return self._mode

    @property
    def project_root(self) -> Path:
        self._require_open()
        return self._loaded.project_root

    @property
    def session_id(self) -> str | None:
        self._require_open()
        return self._writer_lock.session_id if self._writer_lock is not None else None

    @property
    def generation_id(self) -> str:
        self._require_open()
        return self._loaded.generation_id

    @property
    def loaded_project(self) -> LoadedProject:
        self._require_open()
        return self._loaded

    @property
    def project(self) -> ProjectContract:
        self._require_open()
        return self._loaded.project

    @property
    def source_sidecar(self):
        self._require_open()
        return self._loaded.source_sidecar

    @property
    def tower_registry(self):
        self._require_open()
        return self._loaded.tower_registry

    @property
    def span_registry(self):
        self._require_open()
        return self._loaded.span_registry

    @property
    def run_index(self):
        self._require_open()
        return self._loaded.run_index

    @property
    def candidate_store(self):
        self._require_open()
        return self._loaded.candidate_store

    @property
    def tower_group_registry(self):
        self._require_open()
        return self._loaded.tower_group_registry

    @property
    def generation_manifest(self):
        self._require_open()
        return self._loaded.generation_manifest

    @property
    def source_availability(self) -> SourceAvailability:
        self._require_open()
        return self._source_availability

    @property
    def source_path(self) -> Path:
        self._require_open()
        return _source_path(self._loaded)

    @property
    def closed(self) -> bool:
        return self._closed

    def refresh_source_availability(self) -> SourceAvailability:
        """Refresh only fast path/stat/header availability, never a full hash."""

        self._require_open()
        availability, signature = _fast_source_availability(self._loaded)
        if self._strong_cache is not None and signature != self._strong_cache.signature:
            self._strong_cache = None
        elif (
            self._strong_cache is not None
            and availability is SourceAvailability.SOURCE_PRESENT_UNVERIFIED
        ):
            availability = SourceAvailability.SOURCE_VERIFIED
        self._last_source_signature = signature
        self._source_availability = availability
        return availability

    def require_strong_source_identity(self) -> None:
        """Perform or reuse this session's strict source-file digest check.

        A later scientific provider must still perform its own final TOCTOU
        validation around point streaming.  This gate only authorizes a P2B
        session to attempt source-dependent work.
        """

        self._require_open()
        availability = self.refresh_source_availability()
        if availability is SourceAvailability.SOURCE_MISSING:
            raise SourceMissingError("persisted project source is missing")
        if availability is SourceAvailability.SOURCE_IDENTITY_MISMATCH:
            raise SourceIdentityMismatchError("source fails fast identity invariants")

        assert self._last_source_signature is not None
        if (
            self._strong_cache is not None
            and self._strong_cache.signature == self._last_source_signature
        ):
            self._source_availability = SourceAvailability.SOURCE_VERIFIED
            return

        identity = self._loaded.project.source_reference.source_identity
        try:
            validate_source_content_identity_digest(identity)
            validate_source_identity_record_digest(identity)
            resolved_path = Path(self._last_source_signature.resolved_path)
            actual_sha256 = _stream_sha256(resolved_path)
            _, final_signature = _file_signature(self._loaded)
        except FileNotFoundError as exc:
            self._strong_cache = None
            self._source_availability = SourceAvailability.SOURCE_MISSING
            raise SourceMissingError("source disappeared during strong verification") from exc
        except OSError as exc:
            self._strong_cache = None
            self._source_availability = SourceAvailability.SOURCE_IDENTITY_MISMATCH
            raise StrongSourceVerificationError("source cannot be read safely") from exc
        except ValueError as exc:
            self._strong_cache = None
            self._source_availability = SourceAvailability.SOURCE_IDENTITY_MISMATCH
            raise StrongSourceVerificationError("persisted SourceIdentity is invalid") from exc

        if final_signature != self._last_source_signature:
            self._strong_cache = None
            self._last_source_signature = final_signature
            self._source_availability = SourceAvailability.SOURCE_IDENTITY_MISMATCH
            raise StrongSourceVerificationError("source changed during strong verification")
        if actual_sha256 != identity.source_file_sha256:
            self._strong_cache = None
            self._source_availability = SourceAvailability.SOURCE_IDENTITY_MISMATCH
            raise SourceIdentityMismatchError("source file SHA-256 does not match SourceIdentity")

        self._strong_cache = _StrongVerificationCache(final_signature)
        self._last_source_signature = final_signature
        self._source_availability = SourceAvailability.SOURCE_VERIFIED

    def close(self) -> None:
        """Close the session and release writer authority, if any."""

        if self._closed:
            return
        self._closed = True
        self._strong_cache = None
        writer_lock = self._writer_lock
        self._writer_lock = None
        if writer_lock is not None:
            writer_lock.close()

    def reopen(self) -> ProjectSession:
        """Explicitly close this immutable snapshot and open a fresh session."""

        self._require_open()
        root = self._loaded.project_root
        mode = self._mode
        application_version = self._application_version
        self.close()
        if mode is ProjectSessionMode.WRITABLE:
            return open_project_writable(root, application_version=application_version)
        return open_project_read_only(root, application_version=application_version)

    def __enter__(self) -> ProjectSession:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


class WritableProjectSession(ProjectSession):
    """Project session holding one OS writer lock for its full lifetime."""

    def _begin_registry_mutation(self) -> LoadedProject:
        """Reload the authoritative generation while retaining this writer lock."""

        self._require_open()
        if type(self._writer_lock) is not ProjectWriterLock:
            raise ProjectSessionError("writable writer authority is not active")
        try:
            self._writer_lock.assert_live_writer_authority(self._loaded.project_root)
        except Exception as exc:
            raise ProjectSessionError("writable writer authority is not active") from exc
        # A locked mutation always starts from the current pointer.  This also
        # closes the stale-snapshot race for callers that retained this object
        # while another internal publisher advanced the project.
        loaded = persistence.open_project(self._writer_lock.project_root)
        self._accept_published_generation(loaded)
        return loaded

    def _accept_published_generation(self, loaded: LoadedProject) -> None:
        self._require_open()
        if type(self._writer_lock) is not ProjectWriterLock:
            raise ProjectSessionError("cannot refresh a session without writer authority")
        try:
            self._writer_lock.assert_live_writer_authority(loaded.project_root)
        except Exception as exc:
            raise ProjectSessionError("cannot refresh a session without writer authority") from exc
        if loaded.project_root != self._writer_lock.project_root:
            raise ProjectSessionError("published generation belongs to another project root")
        self._loaded = loaded
        self._strong_cache = None
        self._source_availability, self._last_source_signature = _fast_source_availability(loaded)

    def _revalidate_authoritative_state(self) -> LoadedProject:
        """Reopen the active pointer under the still-held writer lock."""

        return self._begin_registry_mutation()

    def reserve_tower_id(self) -> str:
        from core.project.registry import reserve_tower_id

        return reserve_tower_id(self)

    def create_tower(self, candidate, *, reservation: str | None = None):
        from core.project.registry import create_tower

        return create_tower(self, candidate, reservation=reservation)

    def revise_tower(self, tower_id: str, candidate):
        from core.project.registry import revise_tower

        return revise_tower(self, tower_id, candidate)

    def archive_tower(self, tower_id: str, *, cascade_dependent_spans: bool = False):
        from core.project.registry import archive_tower

        return archive_tower(
            self, tower_id, cascade_dependent_spans=cascade_dependent_spans
        )

    def reserve_span_id(self) -> str:
        from core.project.registry import reserve_span_id

        return reserve_span_id(self)

    def create_span(self, candidate, *, reservation: str | None = None):
        from core.project.registry import create_span

        return create_span(self, candidate, reservation=reservation)

    def archive_span(self, span_id: str):
        from core.project.registry import archive_span

        return archive_span(self, span_id)

    def relocate_source(self, candidate_path):
        """Exact-content source relocation; see ``source_relocation``."""
        from core.project.source_relocation import relocate_source

        return relocate_source(self, candidate_path)

    def persist_tower_candidate_set(self, candidate_set):
        from core.project.tower_candidates import persist_tower_candidate_set

        return persist_tower_candidate_set(self, candidate_set)

    def append_tower_candidate_review(
        self,
        *,
        candidate_set_reference,
        candidate_id,
        decision,
        occurred_at,
        reason,
        refinement_state,
        refined_raw_geometry=None,
        refinement_evidence=None,
        operator_identity=None,
        review_adjusted_seed_xyz_absolute_m=None,
        notes=None,
    ):
        from core.project.tower_candidates import append_tower_candidate_review

        return append_tower_candidate_review(
            self,
            candidate_set_reference=candidate_set_reference,
            candidate_id=candidate_id,
            decision=decision,
            occurred_at=occurred_at,
            reason=reason,
            refinement_state=refinement_state,
            refined_raw_geometry=refined_raw_geometry,
            refinement_evidence=refinement_evidence,
            operator_identity=operator_identity,
            review_adjusted_seed_xyz_absolute_m=review_adjusted_seed_xyz_absolute_m,
            notes=notes,
        )

    def confirm_reviewed_tower_candidate(
        self,
        *,
        candidate_set_reference,
        candidate_id,
        accepted_review_revision,
        reviewer_identity=None,
        notes=None,
        confirmed_at=None,
        local_region_reference=None,
        tower_group_registry_updater=None,
    ):
        from core.project.tower_candidates import confirm_reviewed_tower_candidate

        return confirm_reviewed_tower_candidate(
            self,
            candidate_set_reference=candidate_set_reference,
            candidate_id=candidate_id,
            accepted_review_revision=accepted_review_revision,
            reviewer_identity=reviewer_identity,
            notes=notes,
            confirmed_at=confirmed_at,
            local_region_reference=local_region_reference,
            tower_group_registry_updater=tower_group_registry_updater,
        )

    def ensure_ordered_group(self):
        from core.project.tower_groups import ensure_ordered_group

        return ensure_ordered_group(self)

    def create_tower_group(self, *, display_name):
        from core.project.tower_groups import create_tower_group

        return create_tower_group(self, display_name=display_name)

    def append_confirmed_tower_to_group(self, *, group_id, tower_id):
        from core.project.tower_groups import append_confirmed_tower_to_group

        return append_confirmed_tower_to_group(
            self, group_id=group_id, tower_id=tower_id
        )

    def register_run(
        self, run_id: str, span_id: str, *, artifact_root_reference: str | None = None
    ):
        from core.project.run_index import register_run

        return register_run(
            self,
            run_id,
            span_id,
            artifact_root_reference=artifact_root_reference,
        )

    def start_run(self, run_id: str):
        from core.project.run_index import start_run

        return start_run(self, run_id)

    def complete_run(self, run_id: str):
        from core.project.run_index import complete_run

        return complete_run(self, run_id)

    def fail_run(self, run_id: str):
        from core.project.run_index import fail_run

        return fail_run(self, run_id)

    def interrupt_run(self, run_id: str):
        from core.project.run_index import interrupt_run

        return interrupt_run(self, run_id)

    def attach_run_artifact_root(self, run_id: str, artifact_root_reference: str):
        from core.project.run_index import attach_run_artifact_root

        return attach_run_artifact_root(self, run_id, artifact_root_reference)

    def reconcile_run_staleness(self):
        from core.project.run_index import reconcile_run_staleness

        return reconcile_run_staleness(self)


class ReadOnlyProjectSession(ProjectSession):
    """Immutable project snapshot that never acquires writer ownership."""


def _build_writable_session(
    writer_lock: ProjectWriterLock, *, application_version: str
) -> WritableProjectSession:
    try:
        # This is intentionally the first authoritative pointer read.  The
        # lock was acquired before this helper was called.
        loaded = persistence.open_project(writer_lock.project_root)
        return WritableProjectSession(
            loaded,
            mode=ProjectSessionMode.WRITABLE,
            writer_lock=writer_lock,
            application_version=application_version,
            _authority_token=_WRITABLE_SESSION_TOKEN,
        )
    except BaseException:
        try:
            writer_lock.close()
        except BaseException:
            pass
        raise


def open_project_writable(
    project_root: str | os.PathLike[str],
    *,
    application_version: str = DEFAULT_APPLICATION_VERSION,
) -> WritableProjectSession:
    """Open a writable authoritative snapshot after acquiring its writer lock."""

    writer_lock = acquire_project_writer_lock(
        project_root, application_version=application_version
    )
    return _build_writable_session(writer_lock, application_version=application_version)


def open_project_read_only(
    project_root: str | os.PathLike[str],
    *,
    application_version: str = DEFAULT_APPLICATION_VERSION,
) -> ReadOnlyProjectSession:
    """Open exactly one immutable generation without taking writer ownership."""

    root = prepare_project_root(project_root, create_root=False)
    loaded = persistence.open_project(root)
    return ReadOnlyProjectSession(
        loaded,
        mode=ProjectSessionMode.READ_ONLY,
        writer_lock=None,
        application_version=application_version,
    )


def create_project_writable(
    project_root: str | os.PathLike[str],
    project: ProjectContract,
    *,
    source_access_metadata: Mapping[str, Any] | None = None,
    tower_group_registry: Any | None = None,
    application_version: str = DEFAULT_APPLICATION_VERSION,
) -> WritableProjectSession:
    """Create the initial P2A generation while retaining its writer lock."""

    if type(project) is not ProjectContract:
        raise PersistenceError("project must be an exact ProjectContract")
    root = prepare_project_root(project_root, create_root=True)
    writer_lock = acquire_project_writer_lock(root, application_version=application_version)
    try:
        if (root / "project.json").exists():
            raise ProjectAlreadyExistsError("project root already has an active project pointer")
        # P2A's publisher remains intentionally private.  P2B establishes the
        # required lock before calling it and retains the same lock afterward.
        persistence._publish_generation(
            root,
            project,
            source_access_metadata=source_access_metadata,
            tower_group_registry=tower_group_registry,
        )
        return _build_writable_session(writer_lock, application_version=application_version)
    except BaseException:
        try:
            writer_lock.close()
        except BaseException:
            pass
        raise


def reopen_project_writable(
    project_root: str | os.PathLike[str], *, application_version: str = DEFAULT_APPLICATION_VERSION
) -> WritableProjectSession:
    """Explicit named writable reopen convenience API."""

    return open_project_writable(project_root, application_version=application_version)


def reopen_project_read_only(
    project_root: str | os.PathLike[str], *, application_version: str = DEFAULT_APPLICATION_VERSION
) -> ReadOnlyProjectSession:
    """Explicit named read-only reopen convenience API."""

    return open_project_read_only(project_root, application_version=application_version)


__all__ = [
    "ProjectAlreadyExistsError",
    "ProjectSession",
    "ProjectSessionClosedError",
    "ProjectSessionError",
    "ProjectSessionMode",
    "ReadOnlyProjectSession",
    "SourceAvailability",
    "SourceAvailabilityError",
    "SourceFileSignature",
    "SourceIdentityMismatchError",
    "SourceMissingError",
    "StrongSourceVerificationError",
    "WritableProjectSession",
    "create_project_writable",
    "open_project_read_only",
    "open_project_writable",
    "reopen_project_read_only",
    "reopen_project_writable",
]
