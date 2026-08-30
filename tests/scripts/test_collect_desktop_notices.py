from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts.collect_desktop_notices import (
    DEFAULT_PROJ_DATA_ROOT,
    DEFAULT_RUNTIME_ROOT,
    DEFAULT_VENV_ROOT,
    _license_paths,
    _license_text_sections,
    _metadata,
    _native_license_text_sections,
    _native_records,
    _sha256,
    collect_inventory,
    collect_python_distributions,
    render_notices,
)


def test_license_lookup_recurses_declared_dist_info_tree(tmp_path: Path) -> None:
    dist_info = tmp_path / "Lib" / "site-packages" / "demo-1.0.dist-info"
    nested = dist_info / "licenses" / "vendor" / "NOTICE.txt"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"fixture notice\r\n")
    (dist_info / "METADATA").write_text(
        "Name: demo\nVersion: 1.0\nLicense-Expression: MIT\n"
        "License-File: licenses/vendor/NOTICE.txt\n",
        encoding="utf-8",
    )

    paths = _license_paths(dist_info, _metadata(dist_info))

    assert paths == [nested]
    assert _sha256(nested) == hashlib.sha256(b"fixture notice\r\n").hexdigest()


def test_selected_environment_contains_recursive_numpy_and_lazrs_notices() -> None:
    records = collect_python_distributions(DEFAULT_VENV_ROOT)
    by_name = {item["requested_name"]: item for item in records}

    numpy_paths = {item["path"] for item in by_name["numpy"]["license_files"]}
    assert any(path.endswith("numpy/_core/src/highway/LICENSE") for path in numpy_paths)
    assert by_name["lazrs"]["version"] == "0.8.2"
    assert by_name["lazrs"]["license_expression"] == "MIT"
    assert by_name["lazrs"]["license_files"]


def test_native_component_patterns_use_selected_runtime_root(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    payload_path = runtime_root / "Library" / "bin" / "api-ms-win-crt-test.dll"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_bytes(b"native")
    payload = {
        "files": [
            {
                "destination_relative": "Library/bin/api-ms-win-crt-test.dll",
                "path": str(payload_path),
                "file": {"status": "PRESENT"},
            }
        ]
    }

    records = _native_records(payload, runtime_root)
    universal_crt = next(item for item in records if item["name"] == "Universal CRT API-set files")

    assert universal_crt["payload_files"]
    assert universal_crt["payload_files"][0]["path"] == str(payload_path)
    assert json.dumps(records, sort_keys=True)


def test_selected_notice_inventory_records_native_proj_snapshot_and_byte_matches() -> None:
    inventory = collect_inventory(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        proj_data_root=DEFAULT_PROJ_DATA_ROOT,
    )

    assert inventory["native"]["proj_data"]["status"] == "NATIVE_PROJ_DATABASE_PRESENT"
    assert (
        inventory["native"]["proj_data"]["snapshot"]["manifest_payload"]["package"][
            "version"
        ]
        == "9.8.1"
    )
    assert inventory["native"]["conda_matches"]["matched_count"] == 68
    assert all(
        item["status"] == "LOCAL_CACHED_CONDA_PACKAGE_BYTE_MATCH"
        for item in inventory["native"]["components"]
    )
    sqlite_gap = next(
        item
        for item in inventory["unresolved_provenance"]
        if item["status"] == "LICENSE_TEXT_NOT_FOUND_LOCALLY"
        and item["artifact"].startswith("conda:libsqlite")
    )
    assert "blessing" in sqlite_gap["reason"]
    assert "public-domain" in sqlite_gap["reason"]
    assert "not legal clearance" in sqlite_gap["reason"]


def _minimal_native_license_inventory(
    records: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "release_inputs": {},
        "python": {
            "distributions": [],
            "version": "3.11.fixture",
            "license": {"status": "MISSING", "path": "missing-python-license"},
        },
        "browser": {"assets": [], "components": []},
        "native": {
            "manifest": {"path": "fixture-manifest.json", "sha256": "fixture"},
            "conda_matches": {
                "path": "fixture-audit.json",
                "status": "PRESENT",
                "matched_count": 1,
                "packages": records,
            },
            "versions": {},
            "proj_data": {"snapshot": {}},
            "optional_plugin_policy": None,
            "files": [],
            "components": [],
        },
        "webview2_prerequisite": {
            "installer": {"path": "missing-installer", "status": "MISSING"},
            "authenticode_record": {"path": "missing-auth"},
            "eula_record": {"path": "missing-eula"},
            "status": "PREREQUISITE_NOT_EXECUTED",
            "clean_pc_verified": False,
        },
        "unresolved_provenance": [],
    }


def test_render_includes_exact_cached_native_license_bytes_and_preserves_identities(
    tmp_path: Path,
) -> None:
    path = tmp_path / "LICENSE.txt"
    raw = b"cached native license bytes\r\n"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    license_record = {
        "path": str(path),
        "status": "PRESENT",
        "bytes": len(raw),
        "sha256": digest,
    }
    inventory = _minimal_native_license_inventory(
        [
            {
                "name": "fixture-a",
                "version": "1.0",
                "build": "h0",
                "package_root": str(tmp_path / "fixture-a"),
                "license_files": [dict(license_record)],
            },
            {
                "name": "fixture-b",
                "version": "2.0",
                "build": "h1",
                "package_root": str(tmp_path / "fixture-b"),
                "license_files": [dict(license_record)],
            },
        ]
    )

    sections = _license_text_sections(inventory)
    rendered = render_notices(inventory)

    assert len(sections) == 1
    assert "conda:fixture-a 1.0 h0" in sections[0][0]
    assert "conda:fixture-b 2.0 h1" in sections[0][0]
    assert sections[0][2] == raw
    assert raw in rendered
    assert b"conda:fixture-a 1.0 h0 / conda:fixture-b 2.0 h1" in rendered


@pytest.mark.parametrize(
    ("record", "expected_status"),
    [
        (
            {
                "path": "missing-license.txt",
                "status": "MISSING_LOCAL_ARTIFACT",
                "bytes": None,
                "sha256": None,
            },
            "MISSING_LOCAL_ARTIFACT",
        ),
        (
            {
                "path": "placeholder-license.txt",
                "status": "PRESENT",
                "bytes": 999,
                "sha256": "0" * 64,
            },
            "LICENSE_TEXT_HASH_MISMATCH",
        ),
    ],
)
def test_cached_native_license_gap_is_explicitly_unresolved(
    tmp_path: Path,
    record: dict[str, object],
    expected_status: str,
) -> None:
    if record["path"] == "placeholder-license.txt":
        path = tmp_path / str(record["path"])
        path.write_bytes(b"changed license bytes")
        record = {**record, "path": str(path)}
    inventory = _minimal_native_license_inventory(
        [
            {
                "name": "fixture",
                "version": "1.0",
                "build": "h0",
                "package_root": str(tmp_path / "fixture"),
                "license_files": [record],
            }
        ]
    )

    sections, unresolved = _native_license_text_sections(inventory)

    assert sections == []
    assert any(item["status"] == expected_status for item in unresolved)
