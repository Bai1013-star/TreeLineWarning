from __future__ import annotations

import errno
import os
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from core.project import (
    ProjectLockBusy,
    ProjectSessionMode,
    create_project_writable,
    open_project_read_only,
    open_project_writable,
)
from core.project import locking
from core.project.locking import LockMetadata, acquire_project_writer_lock, read_lock_metadata
from core.project.locking import ProjectLockAcquisitionError
from core.project.persistence import PersistenceError


def _create_project(root: Path, project, metadata) -> None:
    session = create_project_writable(root, project, source_access_metadata=metadata)
    session.close()


def _child_writable_attempt(root: Path) -> subprocess.CompletedProcess[str]:
    script = """
import sys
from core.project import ProjectLockBusy, open_project_writable
try:
    session = open_project_writable(sys.argv[1])
except ProjectLockBusy:
    print('BUSY')
else:
    print('ACQUIRED')
    session.close()
"""
    return subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )


def test_windows_winerror_takes_priority_over_errno_for_busy_classification():
    assert locking._is_busy_error(OSError(errno.EACCES, "lock violation", None, 33))
    assert not locking._is_busy_error(OSError(errno.EACCES, "access denied", None, 5))
    assert not locking._is_busy_error(OSError(errno.EIO, "disk full", None, 112))


def test_busy_errno_fallback_is_preserved_without_windows_specific_detail():
    assert locking._is_busy_error(OSError(errno.EACCES, "portable contention"))
    assert locking._is_busy_error(OSError(errno.EAGAIN, "portable contention"))


def test_windows_permission_failure_is_lock_acquisition_error_not_busy(tmp_path, monkeypatch):
    path = tmp_path / "project.lock"
    path.write_bytes(b"0")

    def deny_lock(*args, **kwargs):
        raise OSError(errno.EACCES, "access denied", None, 5)

    fake_msvcrt = types.SimpleNamespace(LK_NBLCK=1, locking=deny_lock)
    monkeypatch.setattr(locking.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    with path.open("r+b") as handle:
        with pytest.raises(ProjectLockAcquisitionError):
            locking._acquire_os_lock(handle)


def test_one_writer_acquires_and_second_writer_does_not_silently_downgrade(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)

    writer = open_project_writable(root)
    assert writer.mode is ProjectSessionMode.WRITABLE
    with pytest.raises(ProjectLockBusy):
        open_project_writable(root)
    writer.close()


def test_second_writer_in_a_separate_process_is_rejected(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    writer = open_project_writable(root)
    try:
        attempted = _child_writable_attempt(root)
        assert attempted.returncode == 0, attempted.stderr
        assert attempted.stdout.strip() == "BUSY"
    finally:
        writer.close()
    reacquired = _child_writable_attempt(root)
    assert reacquired.returncode == 0, reacquired.stderr
    assert reacquired.stdout.strip() == "ACQUIRED"


def test_close_and_context_manager_release_writer_lock(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)

    with open_project_writable(root) as writer:
        first_session_id = writer.session_id
        assert first_session_id is not None
    second = open_project_writable(root)
    assert second.session_id != first_session_id
    second.close()


def test_failed_writable_open_releases_its_lock(tmp_path):
    root = tmp_path / "not-a-project"
    root.mkdir()
    with pytest.raises(PersistenceError):
        open_project_writable(root)
    lock = acquire_project_writer_lock(root)
    lock.close()


def test_stale_and_malformed_free_metadata_are_recovered_after_os_lock_acquisition(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    lock_path = root / "project.lock"
    lock_path.write_bytes(b"malformed stale lock metadata")

    writer = open_project_writable(root)
    try:
        recovered_session_id = writer.session_id
    finally:
        writer.close()
    metadata = read_lock_metadata(root)
    assert metadata.session_id == recovered_session_id
    assert metadata.process_id == os.getpid()


def test_old_timestamp_and_reused_pid_never_override_live_os_lock(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    writer = open_project_writable(root)
    try:
        writer_lock = writer._writer_lock
        assert writer_lock is not None and writer_lock._handle is not None
        locking._write_metadata(
            writer_lock._handle,
            LockMetadata(
                schema_version=locking.LOCK_METADATA_SCHEMA_VERSION,
                session_id=writer.session_id,
                process_id=os.getpid(),
                hostname="misleading-owner",
                opened_at=datetime(2000, 1, 1, tzinfo=UTC).isoformat().replace("+00:00", "Z"),
                application_version="old-version",
            ),
        )
        attempted = _child_writable_attempt(root)
        assert attempted.returncode == 0, attempted.stderr
        assert attempted.stdout.strip() == "BUSY"
    finally:
        writer.close()


def test_malformed_live_metadata_does_not_permit_takeover(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    writer = open_project_writable(root)
    try:
        writer_lock = writer._writer_lock
        assert writer_lock is not None and writer_lock._handle is not None
        writer_lock._handle.seek(1)
        writer_lock._handle.write(b"\xff\xff\xff\xff")
        writer_lock._handle.flush()
        os.fsync(writer_lock._handle.fileno())
        attempted = _child_writable_attempt(root)
        assert attempted.returncode == 0, attempted.stderr
        assert attempted.stdout.strip() == "BUSY"
    finally:
        writer.close()


def test_read_only_open_is_allowed_while_writer_is_active_and_owns_no_writer_lock(
    tmp_path, tiny_project, deterministic_source_metadata
):
    root = tmp_path / "project"
    _create_project(root, tiny_project, deterministic_source_metadata)
    writer = open_project_writable(root)
    reader = open_project_read_only(root)
    try:
        assert reader.mode is ProjectSessionMode.READ_ONLY
        assert reader.session_id is None
    finally:
        writer.close()

    # A live read-only snapshot cannot block a new writer.
    second_writer = open_project_writable(root)
    second_writer.close()
    reader.close()


def test_lock_path_directory_fails_closed(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "project.lock").mkdir()
    with pytest.raises(locking.ProjectLockPathError):
        acquire_project_writer_lock(root)
