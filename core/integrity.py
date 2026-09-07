"""Process-local reuse of cryptographic file validation.

Every digest is still computed at the first trusted boundary.  Reuse is bound
to a strong filesystem snapshot and is discarded when the file is replaced or
its size/timestamps change.  A before/after stat check prevents publishing a
digest for a file that changed while it was read.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

_BLOCK_BYTES = 1 << 20
_MAX_ENTRIES = 1024


class IntegritySnapshot(NamedTuple):
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int


class ArtifactChangedDuringValidation(RuntimeError):
    """The artifact was not immutable for the duration of validation."""


_CACHE: OrderedDict[tuple[str, IntegritySnapshot], str] = OrderedDict()
_LOCK = threading.RLock()
_ACTIVE_SESSIONS = 0


def _snapshot(path: Path) -> IntegritySnapshot:
    stat = path.stat()
    return IntegritySnapshot(
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _hash_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(_BLOCK_BYTES), b""):
        digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str | Path, *, reusable: bool = False) -> str:
    """Return SHA-256, reusing only an unchanged process-local file snapshot."""

    resolved = Path(path).expanduser().resolve()
    before = _snapshot(resolved)
    key = (str(resolved), before)
    with _LOCK:
        session_active = _ACTIVE_SESSIONS > 0
    if not reusable or not session_active:
        with resolved.open("rb") as stream:
            value = _hash_stream(stream)
        if _snapshot(resolved) != before:
            raise ArtifactChangedDuringValidation(
                f"ARTIFACT_CHANGED_DURING_VALIDATION:{resolved}"
            )
        return value

    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached
    with resolved.open("rb") as stream:
        value = _hash_stream(stream)
    if _snapshot(resolved) != before:
        raise ArtifactChangedDuringValidation(
            f"ARTIFACT_CHANGED_DURING_VALIDATION:{resolved}"
        )
    with _LOCK:
        if _ACTIVE_SESSIONS == 0:
            return value
        if _snapshot(resolved) != before:
            raise ArtifactChangedDuringValidation(
                f"ARTIFACT_CHANGED_DURING_VALIDATION:{resolved}"
            )
        stale = [item for item in _CACHE if item[0] == str(resolved) and item != key]
        for item in stale:
            _CACHE.pop(item, None)
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return value


def clear_validation_cache() -> None:
    """Clear process-local validation state (tests and explicit run boundaries)."""

    with _LOCK:
        _CACHE.clear()


@contextmanager
def validation_session() -> Iterator[None]:
    """Bound validated-digest reuse to one or more active pipeline runs."""

    global _ACTIVE_SESSIONS
    with _LOCK:
        _ACTIVE_SESSIONS += 1
    try:
        yield
    finally:
        with _LOCK:
            _ACTIVE_SESSIONS -= 1
            if _ACTIVE_SESSIONS == 0:
                _CACHE.clear()


__all__ = [
    "ArtifactChangedDuringValidation",
    "IntegritySnapshot",
    "clear_validation_cache",
    "sha256_file",
    "validation_session",
]
