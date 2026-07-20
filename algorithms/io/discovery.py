"""Read-only LAS/LAZ file discovery."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from core.contracts import PointCloudSource

POINT_CLOUD_SUFFIXES = {".las", ".laz"}


def discover_las_sources(raw_root: Path | str) -> list[PointCloudSource]:
    """Recursively discover LAS/LAZ files under ``raw_root`` without opening them."""
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"LAS raw root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"LAS raw root must be a directory: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in POINT_CLOUD_SUFFIXES
    ]
    files.sort(key=lambda path: path.relative_to(root).as_posix().lower())

    return [
        PointCloudSource(
            source_id=stable_source_id(path, root),
            path=path,
            role="raw",
            size_bytes=path.stat().st_size,
            readonly_required=True,
        )
        for path in files
    ]


def stable_source_id(path: Path | str, root: Path | str) -> str:
    """Return a deterministic, unique-enough raw source id from a relative path."""
    file_path = Path(path)
    root_path = Path(root)
    try:
        relative = file_path.relative_to(root_path)
    except ValueError:
        relative = file_path
    normalized = relative.as_posix().lower()
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"raw_{slug}_{digest}"
