"""OS-backed project single-writer locking for P2B project sessions.

``project.lock`` has two deliberately separate parts: byte zero is a stable
one-byte OS-lock region and the remaining fixed-size slot holds diagnostic
metadata.  The lock byte is never truncated or rewritten after it has been
created.  This matters on Windows, where ``msvcrt.locking`` requires a
non-empty range and the lock is associated with the open file handle.

Metadata is advisory only.  A writer owns a project only after the operating
system grants the non-blocking exclusive byte-range lock.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import struct
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


LOCK_METADATA_SCHEMA_VERSION = "treeline-project-lock-v1"
DEFAULT_APPLICATION_VERSION = "TreeLineWarning-P2B"

_LOCK_BYTE_OFFSET = 0
_LOCK_LENGTH = 1
_LOCK_SLOT_SIZE = 4096
_METADATA_LENGTH_OFFSET = 1
_METADATA_OFFSET = 5
_MAX_METADATA_BYTES = _LOCK_SLOT_SIZE - _METADATA_OFFSET
# ``msvcrt.locking`` is a byte-range lock operation.  If it preserves a
# Windows-specific code, ERROR_LOCK_VIOLATION is the applicable conflict.  A
# sharing violation belongs to file opening, which is handled separately.
_WINDOWS_LOCK_CONFLICT_WINERRORS = frozenset({33})
_PROCESS_LOCKS: set[str] = set()
_PROCESS_LOCKS_GUARD = threading.Lock()


class ProjectLockError(RuntimeError):
    """Base error for P2B project-lock failures."""


class ProjectLockPathError(ProjectLockError):
    """Raised when ``project.lock`` is not a safe file inside the project."""


class ProjectLockAcquisitionError(ProjectLockError):
    """Raised when the OS lock cannot be acquired because of an I/O failure."""


class ProjectLockBusy(ProjectLockError):
    """Raised when another process currently owns the OS writer lock."""

    def __init__(self, project_root: Path, diagnostic: "LockDiagnostic") -> None:
        super().__init__(f"project already has an active writer: {project_root}")
        self.project_root = project_root
        self.diagnostic = diagnostic


class ProjectLockDiagnosticError(ProjectLockError):
    """Raised only when callers explicitly request strict lock metadata."""


@dataclass(frozen=True, slots=True)
class LockMetadata:
    """Strictly validated, non-authoritative writer diagnostic metadata."""

    schema_version: str
    session_id: str
    process_id: int
    hostname: str
    opened_at: str
    application_version: str

    def __post_init__(self) -> None:
        if self.schema_version != LOCK_METADATA_SCHEMA_VERSION:
            raise ValueError("unsupported project-lock metadata schema version")
        if type(self.session_id) is not str or str(uuid.UUID(self.session_id)) != self.session_id:
            raise ValueError("session_id must be a canonical UUID")
        if type(self.process_id) is not int or self.process_id <= 0:
            raise ValueError("process_id must be a positive integer")
        _bounded_text(self.hostname, "hostname", maximum=255)
        _utc_timestamp(self.opened_at, "opened_at")
        _bounded_text(self.application_version, "application_version", maximum=128)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "process_id": self.process_id,
            "hostname": self.hostname,
            "opened_at": self.opened_at,
            "application_version": self.application_version,
        }


@dataclass(frozen=True, slots=True)
class LockDiagnostic:
    """Safe metadata result attached to a busy-lock error."""

    metadata: LockMetadata | None
    metadata_error: str | None


def _bounded_text(value: object, field_name: str, *, maximum: int) -> None:
    if type(value) is not str or not value or len(value) > maximum:
        raise ValueError(f"{field_name} must be a non-empty bounded string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")


def _utc_timestamp(value: object, field_name: str) -> None:
    if type(value) is not str or not value.endswith("Z") or "T" not in value:
        raise ValueError(f"{field_name} must be a UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid UTC timestamp") from exc
    if parsed.tzinfo != UTC:
        raise ValueError(f"{field_name} must be a UTC timestamp")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _safe_project_root_and_lock_path(
    project_root: str | os.PathLike[str],
    *,
    create_root: bool,
) -> tuple[Path, Path]:
    root = Path(project_root)
    if create_root:
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProjectLockPathError(f"cannot create project root: {root}") from exc
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ProjectLockPathError(f"project root cannot be resolved: {root}") from exc
    if not resolved_root.is_dir():
        raise ProjectLockPathError("project root must be an existing directory")

    lock_path = resolved_root / "project.lock"
    try:
        resolved_lock = lock_path.resolve(strict=False)
    except OSError as exc:
        raise ProjectLockPathError("project.lock cannot be resolved safely") from exc
    if not _is_within(resolved_lock, resolved_root):
        raise ProjectLockPathError("project.lock resolves outside the selected project root")
    if lock_path.exists() and lock_path.is_dir():
        raise ProjectLockPathError("project.lock must be a regular file, not a directory")
    return resolved_root, lock_path


def prepare_project_root(
    project_root: str | os.PathLike[str], *, create_root: bool = False
) -> Path:
    """Validate a project root enough to safely locate its writer lock."""

    root, _ = _safe_project_root_and_lock_path(project_root, create_root=create_root)
    return root


def _open_lock_handle(path: Path) -> BinaryIO:
    try:
        handle = path.open("r+b")
    except FileNotFoundError:
        try:
            handle = path.open("x+b")
        except FileExistsError:
            handle = path.open("r+b")
        except OSError as exc:
            raise ProjectLockAcquisitionError(f"cannot create project.lock: {path}") from exc
    except OSError as exc:
        raise ProjectLockAcquisitionError(f"cannot open project.lock: {path}") from exc

    try:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size < _LOCK_SLOT_SIZE:
            # The first byte is the non-empty lock range required by msvcrt;
            # once present, future rewrites never touch it or truncate the file.
            handle.seek(size)
            handle.write(b"\0" * (_LOCK_SLOT_SIZE - size))
            handle.flush()
            os.fsync(handle.fileno())
        return handle
    except OSError as exc:
        try:
            handle.close()
        finally:
            raise ProjectLockAcquisitionError("cannot prepare project.lock lock range") from exc


def _is_busy_error(error: OSError) -> bool:
    winerror = getattr(error, "winerror", None)
    if winerror is not None:
        # A meaningful Windows code has priority.  In this runtime msvcrt may
        # report a contention as EACCES with no winerror, so errno remains the
        # portable fallback only when Windows-specific detail is unavailable.
        return winerror in _WINDOWS_LOCK_CONFLICT_WINERRORS
    return error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}


def _acquire_os_lock(handle: BinaryIO) -> None:
    try:
        handle.seek(_LOCK_BYTE_OFFSET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _LOCK_LENGTH)
        else:
            import fcntl

            fcntl.lockf(
                handle.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
                _LOCK_LENGTH,
                _LOCK_BYTE_OFFSET,
                os.SEEK_SET,
            )
    except OSError as exc:
        if _is_busy_error(exc):
            raise ProjectLockBusyError from exc
        raise ProjectLockAcquisitionError("OS writer-lock acquisition failed") from exc


class ProjectLockBusyError(Exception):
    """Private sentinel used before a safe busy diagnostic is assembled."""


def _release_os_lock(handle: BinaryIO) -> None:
    try:
        handle.seek(_LOCK_BYTE_OFFSET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _LOCK_LENGTH)
        else:
            import fcntl

            fcntl.lockf(
                handle.fileno(), fcntl.LOCK_UN, _LOCK_LENGTH, _LOCK_BYTE_OFFSET, os.SEEK_SET
            )
    except OSError as exc:
        raise ProjectLockAcquisitionError("OS writer-lock release failed") from exc


def _metadata_payload(metadata: LockMetadata) -> bytes:
    payload = json.dumps(
        metadata.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(payload) > _MAX_METADATA_BYTES:
        raise ProjectLockAcquisitionError("project-lock metadata exceeds its bounded slot")
    return payload


def _write_metadata(handle: BinaryIO, metadata: LockMetadata) -> None:
    payload = _metadata_payload(metadata)
    try:
        # Byte zero remains the lock region.  The fixed-size metadata slot is
        # overwritten in-place, avoiding a truncate/replace that could disturb
        # Windows byte-range lock semantics.
        handle.seek(_METADATA_LENGTH_OFFSET)
        handle.write(struct.pack(">I", len(payload)))
        handle.write(payload)
        handle.write(b"\0" * (_MAX_METADATA_BYTES - len(payload)))
        handle.flush()
        os.fsync(handle.fileno())
    except OSError as exc:
        raise ProjectLockAcquisitionError("cannot durably write project-lock metadata") from exc


def _decode_metadata_bytes(raw: bytes) -> LockMetadata:
    if len(raw) < _METADATA_OFFSET:
        raise ProjectLockDiagnosticError("project-lock metadata slot is truncated")
    length = struct.unpack(">I", raw[_METADATA_LENGTH_OFFSET:_METADATA_OFFSET])[0]
    if length <= 0 or length > _MAX_METADATA_BYTES or _METADATA_OFFSET + length > len(raw):
        raise ProjectLockDiagnosticError("project-lock metadata length is invalid")
    try:
        value = json.loads(raw[_METADATA_OFFSET : _METADATA_OFFSET + length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectLockDiagnosticError("project-lock metadata is malformed") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "session_id",
        "process_id",
        "hostname",
        "opened_at",
        "application_version",
    }:
        raise ProjectLockDiagnosticError("project-lock metadata fields are invalid")
    try:
        return LockMetadata(**value)
    except (TypeError, ValueError) as exc:
        raise ProjectLockDiagnosticError("project-lock metadata values are invalid") from exc


def read_lock_metadata(project_root: str | os.PathLike[str]) -> LockMetadata:
    """Read diagnostic lock metadata strictly; this never establishes ownership."""

    _, path = _safe_project_root_and_lock_path(project_root, create_root=False)
    try:
        with path.open("rb") as handle:
            return _decode_metadata_bytes(handle.read(_LOCK_SLOT_SIZE))
    except FileNotFoundError as exc:
        raise ProjectLockDiagnosticError("project.lock does not exist") from exc
    except OSError as exc:
        raise ProjectLockDiagnosticError("cannot read project-lock metadata") from exc


def _safe_diagnostic(path: Path) -> LockDiagnostic:
    try:
        with path.open("rb") as handle:
            return LockDiagnostic(_decode_metadata_bytes(handle.read(_LOCK_SLOT_SIZE)), None)
    except (OSError, ProjectLockDiagnosticError) as exc:
        return LockDiagnostic(None, str(exc))


def _new_metadata(application_version: str) -> LockMetadata:
    _bounded_text(application_version, "application_version", maximum=128)
    hostname = socket.gethostname()
    _bounded_text(hostname, "hostname", maximum=255)
    return LockMetadata(
        schema_version=LOCK_METADATA_SCHEMA_VERSION,
        session_id=str(uuid.uuid4()),
        process_id=os.getpid(),
        hostname=hostname,
        opened_at=datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z"),
        application_version=application_version,
    )


class ProjectWriterLock:
    """One live OS writer lock; call ``close`` rather than relying on GC."""

    def __init__(
        self,
        *,
        project_root: Path,
        lock_path: Path,
        handle: BinaryIO,
        metadata: LockMetadata,
        process_key: str,
    ) -> None:
        self.project_root = project_root
        self.lock_path = lock_path
        self.metadata = metadata
        self._handle: BinaryIO | None = handle
        self._process_key = process_key

    @property
    def session_id(self) -> str:
        return self.metadata.session_id

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        """Release OS ownership; repeated close calls are safe."""

        handle = self._handle
        if handle is None:
            return
        self._handle = None
        error: BaseException | None = None
        try:
            _release_os_lock(handle)
        except BaseException as exc:  # retain cleanup guarantees below
            error = exc
        try:
            handle.close()
        except BaseException as exc:  # noqa: BLE001 - cleanup path
            if error is None:
                error = exc
        finally:
            with _PROCESS_LOCKS_GUARD:
                _PROCESS_LOCKS.discard(self._process_key)
        if error is not None:
            raise ProjectLockAcquisitionError("failed to close project writer lock") from error

    def __enter__(self) -> "ProjectWriterLock":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def acquire_project_writer_lock(
    project_root: str | os.PathLike[str],
    *,
    application_version: str = DEFAULT_APPLICATION_VERSION,
) -> ProjectWriterLock:
    """Acquire exactly one non-blocking writer lock or raise ``ProjectLockBusy``.

    No authoritative project pointer is read by this function.  Callers must
    acquire this lock before loading or publishing authoritative state.
    """

    root, lock_path = _safe_project_root_and_lock_path(project_root, create_root=False)
    process_key = str(root)
    with _PROCESS_LOCKS_GUARD:
        if process_key in _PROCESS_LOCKS:
            raise ProjectLockBusy(root, LockDiagnostic(None, "writer already active in this process"))
        handle = _open_lock_handle(lock_path)
        try:
            try:
                _acquire_os_lock(handle)
            except ProjectLockBusyError:
                diagnostic = _safe_diagnostic(lock_path)
                raise ProjectLockBusy(root, diagnostic) from None
            metadata = _new_metadata(application_version)
            _write_metadata(handle, metadata)
            _PROCESS_LOCKS.add(process_key)
            return ProjectWriterLock(
                project_root=root,
                lock_path=lock_path,
                handle=handle,
                metadata=metadata,
                process_key=process_key,
            )
        except BaseException:
            try:
                handle.close()
            except OSError:
                pass
            raise


__all__ = [
    "DEFAULT_APPLICATION_VERSION",
    "LOCK_METADATA_SCHEMA_VERSION",
    "LockDiagnostic",
    "LockMetadata",
    "ProjectLockAcquisitionError",
    "ProjectLockBusy",
    "ProjectLockDiagnosticError",
    "ProjectLockError",
    "ProjectLockPathError",
    "ProjectWriterLock",
    "acquire_project_writer_lock",
    "prepare_project_root",
    "read_lock_metadata",
]
