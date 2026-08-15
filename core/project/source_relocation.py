"""Exact-content P2C-B1 source relocation.

Relocation changes only the current audited source access record.  It is
deliberately *not* a scientific or topology mutation:

- the frozen ``source_content_identity_digest`` is preserved exactly;
- the audited access record (``resolved_access_path`` /
  ``recorded_source_path``) changes, so the ``source_identity_record_digest``
  is recomputed through the reviewed FRV record semantics;
- Tower revisions, Span records, the run index, project counters and domain
  identities are preserved byte-for-byte;
- no analysis is marked STALE solely because the filesystem path moved.

Relocation is an authoritative project mutation and therefore requires a
genuine open ``WritableProjectSession`` whose live OS writer capability is
verified by the same unforgeable P2C-A boundary used by registry mutations.
A candidate is only adopted after it strongly verifies to the *frozen expected*
source content identity; the old source file may be missing.

Strong verification here is the *relocation authorization gate*: it proves the
candidate file's content against the frozen identity before the new access
record is published.  It does not replace the later source-provider/scientific
finalization: future scientific execution still performs its own reviewed
source-identity and TOCTOU checks.  "Relocated once" never means "permanently
trusted".
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

from core.contracts.frv import (
    FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
    FrvRelocationAudit,
    SourceIdentity,
    frv_digest,
    validate_source_content_identity_digest,
    validate_source_identity_record_digest,
)
from core.contracts.project import (
    PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
    ProjectSourceReference,
    SourceIdentityBinding,
)
from core.project import persistence
from core.project.persistence import (
    SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION,
    PostPointerDurabilityError,
    SourceReferenceSidecar,
    _utc_timestamp,
)
from core.project.registry import WriterAuthorityError, _updated_project
from core.project.session import WritableProjectSession, _source_path


class SourceRelocationError(RuntimeError):
    """Base error for P2C-B1 exact-content source relocation failures."""


class SourceCandidateError(SourceRelocationError):
    """Raised when the explicitly supplied candidate path is not a safe source."""


class SourceCandidateIdentityError(SourceRelocationError):
    """Raised when a candidate does not strongly verify to the expected identity."""


class SourceRelocationPublicationUncertainError(SourceRelocationError):
    """Raised after the active pointer was replaced but durability is ambiguous."""

    def __init__(self, message: str, *, original: PostPointerDurabilityError) -> None:
        super().__init__(message)
        self.original = original
        self.pointer_replaced = original.pointer_replaced


@dataclass(frozen=True, slots=True)
class SourceRelocationResult:
    """Immutable result of one exact-content source relocation request.

    ``performed`` is ``False`` only for the same-path no-op case, in which case
    no generation was published and every digest/path is the current value.
    """

    performed: bool
    generation_id: str
    source_content_identity_digest: str
    source_identity_record_digest: str
    current_resolved_source_path: str
    previous_resolved_source_path: str | None


def _require_writable(session: WritableProjectSession) -> Any:
    """Reuse the unforgeable P2C-A writable-authority boundary."""
    if type(session) is not WritableProjectSession:
        raise WriterAuthorityError(
            "P2C-B1 relocation requires an exact open WritableProjectSession"
        )
    return session._begin_registry_mutation()


def _validate_candidate_path(candidate_path: str | os.PathLike[str]) -> Path:
    """Validate an explicitly supplied external source candidate path."""
    if not isinstance(candidate_path, str | os.PathLike):
        raise SourceCandidateError("candidate source path must be a string or path-like")
    raw = os.fspath(candidate_path)
    if type(raw) is not str or not raw:
        raise SourceCandidateError("candidate source path must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise SourceCandidateError("candidate source path must not contain control characters")
    path = Path(raw)
    if not path.is_absolute():
        raise SourceCandidateError(
            "candidate source path must be absolute and explicitly supplied"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise SourceCandidateError("candidate source path cannot be resolved") from exc
    if not resolved.is_file():
        raise SourceCandidateError(
            "candidate source path must be a regular file, not a directory or other object"
        )
    if resolved.suffix.lower() not in {".las", ".laz"}:
        raise SourceCandidateError(
            "candidate source path must reference a supported LAS/LAZ source"
        )
    return resolved


def _platform_file_identity(file_stat) -> tuple[int, int] | None:
    """Return the platform file identity used by the reviewed P2B source gate.

    This mirrors ``core.project.session``'s signature identity: ``st_dev`` plus
    ``st_ino`` (on Windows the NTFS volume serial and file index).  Returns
    ``None`` only where the platform/stat object cannot provide an inode.
    """

    device = getattr(file_stat, "st_dev", None)
    inode = getattr(file_stat, "st_ino", None)
    if device is None or inode is None or int(inode) == 0:
        return None
    return (int(device), int(inode))


def _stream_handle_sha256(handle) -> str:
    """Stream a SHA-256 from an already-open stable file handle."""
    digest = hashlib.sha256()
    while True:
        block = handle.read(1 << 20)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


class _VerifiedSourceLease:
    """Internal lease binding a strong-verified candidate source to its path.

    The verified binary handle is intentionally kept open from verification
    through the authoritative publication attempt, so the adopted path cannot be
    swapped to unverified content on the primary Windows target.  ``assert_path_still_bound``
    is the final pre-publication gate; it remains authoritative even if the
    verified handle has been released, because it compares the current path
    identity against the verified file identity captured from the handle that
    was actually hashed.

    This is an internal implementation detail of relocation; it is never exposed
    as a public project API.
    """

    def __init__(
        self,
        handle,
        *,
        resolved_path: Path,
        sha256: str,
        size: int,
        mtime_ns: int,
        platform_identity: tuple[int, int] | None,
    ) -> None:
        self._handle = handle
        self.resolved_path = resolved_path
        self.sha256 = sha256
        self.size = size
        self.mtime_ns = mtime_ns
        self.platform_identity = platform_identity
        self._closed = False

    def assert_path_still_bound(self) -> None:
        """Require the candidate path to still resolve to the verified file.

        Re-identifies the candidate PATH immediately before publication and
        requires it to match the file that was verified (the still-open handle's
        identity where available, otherwise the verified identity captured from
        that handle).  Size/mtime remain defense-in-depth.  On failure relocation
        fails closed with no publication.
        """
        if self._closed:
            verified_identity = self.platform_identity
        else:
            try:
                handle_stat = os.fstat(self._handle.fileno())
            except (OSError, ValueError):
                # The verified handle was externally released; the stored
                # verified identity remains authoritative for this gate.
                verified_identity = self.platform_identity
            else:
                verified_identity = _platform_file_identity(handle_stat)
        current = self.resolved_path.stat()
        current_identity = _platform_file_identity(current)
        if verified_identity is not None and current_identity != verified_identity:
            raise SourceCandidateError(
                "candidate source path no longer resolves to the verified file"
            )
        if (current.st_size, current.st_mtime_ns) != (self.size, self.mtime_ns):
            raise SourceCandidateError("candidate source changed after verification")

    def close(self) -> None:
        """Deterministically release the verified handle; idempotent."""
        if not self._closed:
            self._closed = True
            self._handle.close()

    def __enter__(self) -> "_VerifiedSourceLease":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _open_verified_source_lease(
    candidate: Path, expected: SourceIdentity
) -> _VerifiedSourceLease:
    """Open and strongly verify a candidate source, returning a live lease.

    The candidate's bytes are streamed through one stable handle; the verified
    file identity/size/mtime are captured from that handle (not from a later
    unbound ``candidate.stat()``).  An immediate post-hash path check closes the
    during-hash replacement window, and the returned lease keeps the handle open
    so the final pre-publication binding check can prove the adopted path still
    resolves to the verified file.  The old source path is never required to
    exist, and no point array or scientific algorithm is involved.
    """
    try:
        validate_source_content_identity_digest(expected)
        validate_source_identity_record_digest(expected)
    except ValueError as exc:
        raise SourceCandidateError("persisted expected SourceIdentity is invalid") from exc
    try:
        handle = candidate.open("rb")
    except FileNotFoundError as exc:
        raise SourceCandidateError("candidate source disappeared during verification") from exc
    except OSError as exc:
        raise SourceCandidateError("candidate source cannot be read safely") from exc
    try:
        handle_stat = os.fstat(handle.fileno())
        handle_identity = _platform_file_identity(handle_stat)
        actual_sha256 = _stream_handle_sha256(handle)
    except OSError as exc:
        handle.close()
        raise SourceCandidateError("candidate source cannot be read safely") from exc
    if actual_sha256 != expected.source_file_sha256:
        handle.close()
        raise SourceCandidateIdentityError(
            "candidate source does not match the frozen project source content identity"
        )
    verified = _VerifiedSourceLease(
        handle,
        resolved_path=candidate,
        sha256=actual_sha256,
        size=int(handle_stat.st_size),
        mtime_ns=int(handle_stat.st_mtime_ns),
        platform_identity=handle_identity,
    )
    # Immediate post-hash path check: catches a swap that occurred during hashing
    # while the handle was still being read.
    try:
        verified.assert_path_still_bound()
    except BaseException:
        verified.close()
        raise
    return verified


def _relocated_identity(previous: SourceIdentity, candidate: Path) -> SourceIdentity:
    """Build the new audited SourceIdentity access record.

    The scientific content identity is preserved exactly.  Only the recorded /
    resolved access paths and the audited record digest change, following the
    reviewed FRV relocation semantics.
    """
    resolved = str(candidate)
    lexical = candidate.as_posix()
    is_windows = os.name == "nt"
    audit_fields = {
        "audit_schema_version": FRV_RELOCATION_AUDIT_SCHEMA_VERSION,
        "path_style": "windows" if is_windows else "posix",
        "recorded_path_raw": resolved,
        "resolved_path_raw": resolved,
        "recorded_path_lexical_normalized": lexical,
        "resolved_path_lexical_normalized": lexical,
        "resolved_final_path": resolved,
        "resolution_method": "P2C_B1_EXPLICIT_EXACT_CONTENT_RELOCATION",
        "case_comparison_mode": (
            "WINDOWS_CASE_INSENSITIVE" if is_windows else "CASE_SENSITIVE"
        ),
        "platform_name": "windows" if is_windows else "posix",
        "run_id": f"p2c-b1-relocation-{previous.source_file_sha256[:16]}",
        "audit_timestamp": _utc_timestamp(),
        "content_identity_result": "exact_content_match",
        "warning_flags": 0,
    }
    relocation_audit = FrvRelocationAudit(
        **audit_fields,
        audit_digest=frv_digest(audit_fields),
    )
    content = previous.to_content_digest_dict()
    record = {
        **content,
        "recorded_source_path": resolved,
        "resolved_access_path": resolved,
        "relocation_audit_audit_digest": relocation_audit.audit_digest,
        "source_equivalence_audit_audit_digest": (
            previous.source_equivalence_audit.audit_digest
            if previous.source_equivalence_audit is not None
            else None
        ),
        "source_content_identity_digest": previous.source_content_identity_digest,
    }
    return SourceIdentity(
        source_identity_schema_version=previous.source_identity_schema_version,
        source_id=previous.source_id,
        recorded_source_path=resolved,
        resolved_access_path=resolved,
        source_file_sha256=previous.source_file_sha256,
        canonical_point_content_digest=previous.canonical_point_content_digest,
        canonical_point_content_spec_digest=previous.canonical_point_content_spec_digest,
        canonical_point_content_spec_version=previous.canonical_point_content_spec_version,
        source_point_count=previous.source_point_count,
        source_bounds_min_xyz=previous.source_bounds_min_xyz,
        source_bounds_max_xyz=previous.source_bounds_max_xyz,
        coordinate_frame=previous.coordinate_frame,
        units=previous.units,
        origin_context=previous.origin_context,
        relocation_audit=relocation_audit,
        source_equivalence_audit=previous.source_equivalence_audit,
        source_content_identity_digest=previous.source_content_identity_digest,
        source_identity_record_digest=frv_digest(record),
    )


def _relocated_sidecar(
    sidecar: SourceReferenceSidecar,
    project: Any,
    new_identity: SourceIdentity,
    verified: _VerifiedSourceLease,
) -> SourceReferenceSidecar:
    """Build the candidate source-reference sidecar for the relocated generation.

    The prior SourceIdentity access record is appended to ``source_identity_history``
    so historical Tower/Span bindings and runs remain auditable.  If that record
    is becoming current again (A -> B -> A), it is removed from history so the
    current record digest appears only as CURRENT and history stays deterministic
    and unique.

    Authoritative access metadata comes from the verified lease (the file that
    was actually hashed), never from a later unbound ``candidate.stat()``.
    """
    previous_identity = sidecar.project_source_reference.source_identity
    metadata = dict(sidecar.access_metadata)
    metadata["current_resolved_source_path"] = new_identity.resolved_access_path
    metadata["file_size_bytes"] = verified.size
    metadata["source_available_at_save"] = True
    metadata["recorded_at"] = _utc_timestamp()
    history = (*sidecar.source_identity_history, previous_identity)
    history = tuple(
        item
        for item in history
        if item.source_identity_record_digest
        != new_identity.source_identity_record_digest
    )
    # ``generation_id`` is rebound to the candidate generation by the publisher.
    return SourceReferenceSidecar(
        schema_version=SOURCE_REFERENCE_PERSISTENCE_SCHEMA_VERSION,
        generation_id=sidecar.generation_id,
        project_id=project.project_id,
        project_source_reference=project.source_reference,
        access_metadata=metadata,
        source_identity_history=history,
    )


def relocate_source(
    session: WritableProjectSession,
    candidate_path: str | os.PathLike[str],
) -> SourceRelocationResult:
    """Exact-content source relocation; see module docstring for the contract."""
    loaded = _require_writable(session)

    candidate = _validate_candidate_path(candidate_path)

    current_identity = loaded.project.source_reference.source_identity
    current_source_path = _source_path(loaded)
    if candidate == current_source_path.resolve(strict=False):
        # Same-path relocation: no source access record needs to change, so no
        # meaningless generation is published and no generation number is used.
        return SourceRelocationResult(
            performed=False,
            generation_id=loaded.generation_id,
            source_content_identity_digest=current_identity.source_content_identity_digest,
            source_identity_record_digest=current_identity.source_identity_record_digest,
            current_resolved_source_path=current_identity.resolved_access_path,
            previous_resolved_source_path=None,
        )

    # The verified handle lease stays open through preparation and the final
    # binding check, and is only released when publication succeeds or fails.
    with _open_verified_source_lease(candidate, expected=current_identity) as verified:
        new_identity = _relocated_identity(current_identity, candidate)
        new_binding = SourceIdentityBinding.from_source_identity(new_identity)
        new_source_reference = ProjectSourceReference(
            schema_version=PROJECT_SOURCE_REFERENCE_SCHEMA_VERSION,
            source_identity=new_identity,
            coordinate_context=new_binding.coordinate_context,
        )
        new_project = _updated_project(
            loaded.project,
            source_reference=new_source_reference,
        )
        new_sidecar = _relocated_sidecar(
            loaded.source_sidecar,
            new_project,
            new_identity,
            verified,
        )
        # FINAL path-to-verified-file binding check, immediately before the
        # authoritative publication primitive.  No candidate path access happens
        # between this check and publication.
        verified.assert_path_still_bound()
        try:
            published = persistence._publish_generation(
                loaded.project_root,
                new_project,
                source_sidecar=new_sidecar,
                tower_registry=loaded.tower_registry,
                span_registry=loaded.span_registry,
                run_index=loaded.run_index,
                candidate_store=loaded.candidate_store,
            )
        except PostPointerDurabilityError as exc:
            # Reuse the reviewed P2C-A post-pointer uncertainty handling: the lock
            # is still held, so revalidate the authoritative pointer before
            # reporting the uncertain result.
            try:
                session._revalidate_authoritative_state()
            except BaseException as revalidation_error:
                raise SourceRelocationPublicationUncertainError(
                    "source relocation publication is uncertain and authoritative "
                    f"revalidation failed: {revalidation_error}",
                    original=exc,
                ) from exc
            raise SourceRelocationPublicationUncertainError(
                "source relocation publication is uncertain; authoritative state was "
                "reopened under the writer lock",
                original=exc,
            ) from exc
        session._accept_published_generation(published)
    return SourceRelocationResult(
        performed=True,
        generation_id=published.generation_id,
        source_content_identity_digest=new_identity.source_content_identity_digest,
        source_identity_record_digest=new_identity.source_identity_record_digest,
        current_resolved_source_path=new_identity.resolved_access_path,
        previous_resolved_source_path=current_identity.resolved_access_path,
    )


__all__ = [
    "SourceCandidateError",
    "SourceCandidateIdentityError",
    "SourceRelocationError",
    "SourceRelocationPublicationUncertainError",
    "SourceRelocationResult",
    "relocate_source",
]
