from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.prepare_desktop_proj_data import materialize


def _write_fixture_package(root: Path) -> tuple[Path, Path]:
    source = root / "proj-9.8.1-hd30e2cd_0"
    data_root = source / "Library" / "share" / "proj"
    data_root.mkdir(parents=True)
    (source / "Library" / "bin").mkdir(parents=True)
    (source / "info" / "licenses").mkdir(parents=True)
    database = data_root / "proj.db"
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1])
try:
    connection.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
    connection.executemany(
        "INSERT INTO metadata VALUES (?, ?)",
        [("DATABASE.LAYOUT.VERSION.MAJOR", "1"), ("DATABASE.LAYOUT.VERSION.MINOR", "6")],
    )
    connection.commit()
finally:
    connection.close()
""",
            str(database),
        ],
        check=True,
    )
    (data_root / "proj.ini").write_text("[general]\n", encoding="utf-8")
    dll = source / "Library" / "bin" / "proj_9.dll"
    dll.write_bytes(b"fixture proj dll")
    (source / "info" / "licenses" / "COPYING").write_text(
        "PROJ license fixture\n", encoding="utf-8"
    )
    payload_rows = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file():
            payload_rows.append(
                {
                    "_path": path.relative_to(source).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_in_bytes": path.stat().st_size,
                }
            )
    payload_rows.append(
        {
            "_path": "Library/bin/proj_9.dll",
            "sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
            "size_in_bytes": dll.stat().st_size,
        }
    )
    (source / "info" / "paths.json").write_text(
        json.dumps({"paths": payload_rows}), encoding="utf-8"
    )
    archive = root / "proj-9.8.1-hd30e2cd_0.conda"
    archive.write_bytes(b"fixture conda archive")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    (source / "info" / "index.json").write_text(
        json.dumps(
            {
                "name": "proj",
                "version": "9.8.1",
                "build": "hd30e2cd_0",
                "subdir": "win-64",
                "license": "MIT",
            }
        ),
        encoding="utf-8",
    )
    (source / "info" / "repodata_record.json").write_text(
        json.dumps(
            {
                "name": "proj",
                "version": "9.8.1",
                "build": "hd30e2cd_0",
                "subdir": "win-64",
                "license": "MIT",
                "url": "https://conda.anaconda.org/conda-forge/win-64/"
                "proj-9.8.1-hd30e2cd_0.conda",
                "sha256": archive_hash,
            }
        ),
        encoding="utf-8",
    )
    return source, archive


def test_materialize_binds_payload_to_conda_hash_metadata(tmp_path: Path) -> None:
    source, archive = _write_fixture_package(tmp_path)
    destination = tmp_path / "snapshot"

    manifest_path = materialize(source, destination, archive=archive)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["package"]["version"] == "9.8.1"
    assert manifest["source_proof"]["source_proj_db"]["layout_minor"] == 6
    assert manifest["provenance_copies"][-1]["destination_relative"] == (
        "provenance/source_proof.json"
    )
    assert (destination / "Library/share/proj/proj.db").is_file()
    assert (destination / "provenance/proj-package.conda").read_bytes() == archive.read_bytes()

    with pytest.raises(SystemExit, match="refusing to replace"):
        materialize(source, destination, archive=archive)
