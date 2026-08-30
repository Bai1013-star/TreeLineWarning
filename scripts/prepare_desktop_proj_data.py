"""Materialize the native PROJ data used by the desktop PDAL runtime.

The bundled PDAL DLL is PROJ 9.8.1.  It cannot safely consume the older
``proj.db`` shipped inside the pyproj wheel, so the release keeps a separate
PROJ data snapshot.  This helper copies only ``Library/share/proj`` into a new
immutable destination and records the exact conda package, metadata, license,
DLL and database hashes used to validate the snapshot.  The source package is
never modified and an existing destination is never replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path(r"E:\Miniforge\pkgs\proj-9.8.1-hd30e2cd_0")
DEFAULT_ARCHIVE = DEFAULT_SOURCE.parent / f"{DEFAULT_SOURCE.name}.conda"
DEFAULT_DESTINATION = Path(".build-pdal-proj-data-final-20260830-02")
_EXPECTED = {
    "name": "proj",
    "version": "9.8.1",
    "build": "hd30e2cd_0",
    "subdir": "win-64",
    "license": "MIT",
}
_EXPECTED_URL = (
    "https://conda.anaconda.org/conda-forge/win-64/"
    "proj-9.8.1-hd30e2cd_0.conda"
)
_REPARSE_POINT = 0x400


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & _REPARSE_POINT)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if not path.is_file() or _is_reparse_point(path):
        raise SystemExit(f"required PROJ artifact is not a regular file: {path}")
    return {
        "path": (
            path.relative_to(relative_to).as_posix()
            if relative_to is not None
            else str(path)
        ),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _ensure_real_tree(root: Path) -> None:
    if not root.is_dir() or _is_reparse_point(root):
        raise SystemExit(f"PROJ data source must be a real directory: {root}")
    for path in root.rglob("*"):
        if _is_reparse_point(path):
            raise SystemExit(f"PROJ data source contains a link/reparse point: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read PROJ package metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"PROJ package metadata must be an object: {path}")
    return value


def _validate_package(
    source: Path, archive: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import sqlite3

    index_path = source / "info" / "index.json"
    repodata_path = source / "info" / "repodata_record.json"
    paths_path = source / "info" / "paths.json"
    license_path = source / "info" / "licenses" / "COPYING"
    data_root = source / "Library" / "share" / "proj"
    database_path = data_root / "proj.db"
    dll_path = source / "Library" / "bin" / "proj_9.dll"
    for key, path in {
        "info/index.json": index_path,
        "info/repodata_record.json": repodata_path,
        "info/paths.json": paths_path,
        "info/licenses/COPYING": license_path,
        "Library/share/proj": data_root,
        "Library/share/proj/proj.db": database_path,
        "Library/bin/proj_9.dll": dll_path,
        "package archive": archive,
    }.items():
        if key == "Library/share/proj":
            _ensure_real_tree(path)
        elif not path.is_file() or _is_reparse_point(path):
            raise SystemExit(f"required PROJ package artifact is missing: {path}")

    index = _load_json(index_path)
    repodata = _load_json(repodata_path)
    paths_metadata = _load_json(paths_path)
    for key, expected in _EXPECTED.items():
        if index.get(key) != expected:
            raise SystemExit(
                f"PROJ package metadata mismatch for {key}: "
                f"expected {expected!r}, found {index.get(key)!r}"
            )
        if repodata.get(key) != expected:
            raise SystemExit(
                f"PROJ repodata mismatch for {key}: "
                f"expected {expected!r}, found {repodata.get(key)!r}"
            )
    archive_hash = _sha256(archive)
    if repodata.get("url") != _EXPECTED_URL:
        raise SystemExit(
            f"PROJ package URL mismatch: expected {_EXPECTED_URL}, found {repodata.get('url')!r}"
        )
    if repodata.get("sha256", "").casefold() != archive_hash.casefold():
        raise SystemExit(
            "PROJ package archive hash does not match info/repodata_record.json: "
            f"{archive_hash}"
        )
    path_rows = {
        str(item.get("_path")): item
        for item in paths_metadata.get("paths", [])
        if isinstance(item, dict) and item.get("_path")
    }
    for payload_path in sorted(data_root.rglob("*")):
        if not payload_path.is_file():
            continue
        relative = payload_path.relative_to(source).as_posix()
        expected = path_rows.get(relative)
        if expected is None:
            raise SystemExit(f"PROJ payload is absent from info/paths.json: {relative}")
        if expected.get("size_in_bytes") != payload_path.stat().st_size:
            raise SystemExit(f"PROJ payload size differs from info/paths.json: {relative}")
        if str(expected.get("sha256", "")).casefold() != _sha256(payload_path).casefold():
            raise SystemExit(f"PROJ payload hash differs from info/paths.json: {relative}")
    for expected_relative in ("Library/share/proj/proj.db", "Library/bin/proj_9.dll"):
        expected = path_rows.get(expected_relative)
        actual_path = source / expected_relative
        if expected is None or str(expected.get("sha256", "")).casefold() != (
            _sha256(actual_path).casefold()
        ):
            raise SystemExit(f"PROJ source hash differs from info/paths.json: {expected_relative}")
    try:
        connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT key, value FROM metadata "
                "WHERE key IN ('DATABASE.LAYOUT.VERSION.MAJOR', 'DATABASE.LAYOUT.VERSION.MINOR')"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise SystemExit(f"cannot inspect PROJ database layout: {exc}") from exc
    try:
        layout = {str(key): int(value) for key, value in rows}
        layout_major = layout.get("DATABASE.LAYOUT.VERSION.MAJOR")
        layout_minor = layout.get("DATABASE.LAYOUT.VERSION.MINOR")
    except (TypeError, ValueError) as exc:
        raise SystemExit("PROJ database has no numeric layout minor") from exc
    if layout_major != 1 or layout_minor is None or layout_minor < 6:
        raise SystemExit(
            "PROJ database layout minor must be at least 6 for native PROJ 9.8.1: "
            f"{layout_minor!r}"
        )
    return (
        index,
        repodata,
        {
            "package_archive": _file_record(archive),
            "license": _file_record(license_path),
            "source_proj_dll": _file_record(dll_path),
            "source_proj_db": {
                **_file_record(database_path),
                "layout_major": layout_major,
                "layout_minor": layout_minor,
            },
            "paths_metadata": _file_record(paths_path),
        },
    )


def _copy_tree(
    source: Path,
    destination: Path,
    source_root: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file() or _is_reparse_point(path):
            raise SystemExit(f"PROJ data payload is not a regular file: {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        rows.append(
            {
                "source_relative": path.relative_to(source_root).as_posix(),
                "destination_relative": target.relative_to(output_root).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )
    return rows


def _copy_provenance(
    source: Path, archive: Path, temporary: Path, proof: dict[str, Any]
) -> list[dict[str, Any]]:
    provenance = temporary / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    copies = {
        "proj-package.conda": archive,
        "info-index.json": source / "info" / "index.json",
        "paths.json": source / "info" / "paths.json",
        "repodata_record.json": source / "info" / "repodata_record.json",
        "COPYING": source / "info" / "licenses" / "COPYING",
        "source-proj_9.dll": source / "Library" / "bin" / "proj_9.dll",
    }
    rows: list[dict[str, Any]] = []
    for name, original in copies.items():
        target = provenance / name
        shutil.copy2(original, target)
        rows.append(
            {
                "source": str(original),
                "destination_relative": target.relative_to(temporary).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
                "matches_source": _sha256(target) == _sha256(original),
            }
        )
    manifest = {
        "package": proof,
        "provenance_copies": rows,
    }
    source_proof = provenance / "source_proof.json"
    source_proof.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    rows.append(
        {
            "source": None,
            "destination_relative": source_proof.relative_to(temporary).as_posix(),
            "bytes": source_proof.stat().st_size,
            "sha256": _sha256(source_proof),
            "matches_source": None,
        }
    )
    return rows


def materialize(
    source: Path, destination: Path, *, archive: Path | None = None
) -> Path:
    """Create and verify a new PROJ data snapshot, returning its manifest."""

    source = source.resolve()
    destination = destination.resolve()
    archive = (archive or source.parent / f"{source.name}.conda").resolve()
    if destination.exists():
        raise SystemExit(f"refusing to replace existing PROJ data snapshot: {destination}")
    index, repodata, proof = _validate_package(source, archive)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=str(destination.parent))
    )
    try:
        data_destination = temporary / "Library" / "share" / "proj"
        payload = _copy_tree(
            source / "Library" / "share" / "proj",
            data_destination,
            source,
            temporary,
        )
        provenance = _copy_provenance(source, archive, temporary, proof)
        manifest = {
            "schema": 1,
            "created_utc": datetime.now(UTC).isoformat(),
            "source_root": str(source),
            "source_archive": str(archive),
            "source_url": _EXPECTED_URL,
            "destination_root": str(destination),
            "snapshot_policy": "new_destination_only; source and pyproj wheel remain unchanged",
            "package": {
                "name": index["name"],
                "version": index["version"],
                "build": index["build"],
                "subdir": index["subdir"],
                "license": index["license"],
                "repodata_url": repodata["url"],
                "repodata_sha256": repodata["sha256"],
            },
            "payload_root": "Library/share/proj",
            "files": payload,
            "source_proof": proof,
            "provenance_copies": provenance,
        }
        manifest_path = temporary / "proj_data_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        _ensure_real_tree(temporary)
        if destination.exists():
            raise SystemExit(f"refusing to replace existing PROJ data snapshot: {destination}")
        os.replace(temporary, destination)
        return destination / "proj_data_manifest.json"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = materialize(args.source, args.destination, archive=args.archive)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
