from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import scripts.build_desktop_release as release


def _write_bytes(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "destination_relative": path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _write_native(root: Path) -> Path:
    rows: list[dict[str, object]] = []
    for relative, payload in (
        (
            "Lib/site-packages/pdal/libpdalpython.cp311-win_amd64.pyd",
            b"binding",
        ),
        ("Library/bin/pdalcpp.dll", b"pdal"),
    ):
        row = _write_bytes(root / relative, payload)
        row["destination_relative"] = relative
        rows.append(row)
    compatibility: list[dict[str, object]] = []
    for relative, payload in (
        ("compat/openjp2.dll", b"openjp2"),
        ("compat/zlib-ng2.dll", b"zlib"),
    ):
        row = _write_bytes(root / relative, payload)
        row["destination_relative"] = relative
        compatibility.append(row)
    (root / "native_runtime_manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "snapshot_policy": "new_destination_only; no junctions or symlinks",
                "versions": {"PDAL": "fixture"},
                "files": rows,
                "compatibility_files": compatibility,
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_proj(root: Path) -> Path:
    payload = _write_bytes(root / "Library/share/proj/proj.db", b"proj database")
    payload["destination_relative"] = "Library/share/proj/proj.db"
    files = [payload]
    provenance_row = _write_bytes(root / "provenance/source.txt", b"source copy")
    provenance_row["destination_relative"] = "provenance/source.txt"
    provenance = [provenance_row]
    proof = _write_bytes(root / "provenance/source_proof.json", b"proof")
    proof["destination_relative"] = "provenance/source_proof.json"
    for row in provenance:
        row["matches_source"] = True
        row["source"] = "fixture-source"
    proof["source"] = None
    proof["matches_source"] = None
    (root / "proj_data_manifest.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "files": files,
                "provenance_copies": [*provenance, proof],
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_python(root: Path) -> Path:
    venv = root / "venv"
    executable = venv / "Scripts" / "python.exe"
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.write_bytes(b"fixture python")
    (venv / "pyvenv.cfg").write_text(
        "version = 3.11.9\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    (venv / "Lib" / "site-packages" / "PyInstaller").mkdir(parents=True)
    return executable


def _write_inputs(root: Path) -> dict[str, Path]:
    notice_text = root / "THIRD_PARTY_NOTICES.txt"
    notice_text.write_text("ENGINEERING INVENTORY ONLY\n", encoding="utf-8")
    notice_json = root / "THIRD_PARTY_NOTICES.json"
    notice_json.write_text(
        json.dumps(
            {
                "inventory_kind": "desktop-third-party-notices",
                "unresolved_provenance": [],
            }
        ),
        encoding="utf-8",
    )
    guide = root / "WINDOWS_USER_GUIDE.zh-CN.md"
    guide.write_text("Survey-only fixture guide\n", encoding="utf-8")
    spec = root / "desktop_build.spec"
    spec.write_text("# fixture spec\n", encoding="utf-8")
    return {
        "native": _write_native(root / "native"),
        "proj": _write_proj(root / "proj"),
        "python": _write_python(root),
        "notice_text": notice_text,
        "notice_json": notice_json,
        "guide": guide,
        "spec": spec,
        "webview": root / "webview",
    }


def _options(root: Path, inputs: dict[str, Path], **overrides: object) -> release.ReleaseOptions:
    values: dict[str, object] = {
        "dist_root": root / "dist",
        "work_root": root / "work",
        "native_runtime": inputs["native"],
        "proj_data_root": None,
        "python_executable": inputs["python"],
        "spec_path": inputs["spec"],
        "guide_path": inputs["guide"],
        "notice_text": inputs["notice_text"],
        "notice_json": inputs["notice_json"],
        "webview_root": inputs["webview"],
        "execute": False,
        "include_webview": False,
        "timeout_seconds": 30.0,
    }
    values.update(overrides)
    return release.ReleaseOptions(**values)


def _stable_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        release,
        "tracked_source_digest",
        lambda: {
            "head": "fixture-head",
            "tracked_source_sha256": "a" * 64,
            "tracked_file_count": 2,
        },
    )


def test_tracked_source_digest_uses_framed_path_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.txt").write_bytes(b"ab")
    (tmp_path / "b.txt").write_bytes(b"c")

    def fake_git(args: list[str], *, binary: bool = False) -> str | bytes:
        if args[0] == "rev-parse":
            return "fixture-head\n"
        assert binary
        return b"a.txt\0b.txt\0"

    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(release, "_git_capture", fake_git)
    first = release.tracked_source_digest()
    second = release.tracked_source_digest()

    assert first == second
    assert first["head"] == "fixture-head"
    assert first["tracked_file_count"] == 2
    assert len(first["tracked_source_sha256"]) == 64


def test_proj_validator_requires_source_bound_provenance(tmp_path: Path) -> None:
    root = _write_proj(tmp_path / "proj")
    validated = release._validate_proj_data(root)
    assert validated is not None
    assert validated.file_count == 3

    manifest_path = root / "proj_data_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["provenance_copies"][0]["matches_source"] = False
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="matched to their source"):
        release._validate_proj_data(root)


def test_webview_validator_binds_auth_receipt_to_offline_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"test"
    monkeypatch.setattr(release, "WEBVIEW2_INSTALLER_BYTES", len(payload))
    monkeypatch.setattr(
        release,
        "WEBVIEW2_INSTALLER_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    root = tmp_path / "webview"
    installer = root / "offline_verified" / release.WEBVIEW2_INSTALLER_NAME
    installer.parent.mkdir(parents=True)
    installer.write_bytes(payload)
    (root / "authenticode.json").write_text(
        json.dumps(
            {
                "path": str(installer),
                "status": "Valid",
                "executed": False,
                "length": len(payload),
            }
        ),
        encoding="utf-8",
    )
    (root / "offline_download.json").write_text(
        json.dumps(
            {
                "action": "download_only_not_executed",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "relative_path": f"offline_verified/{release.WEBVIEW2_INSTALLER_NAME}",
            }
        ),
        encoding="utf-8",
    )
    (root / "official_webview2_eula.json").write_text(
        json.dumps({"evergreenHtml": "<html>fixture</html>"}),
        encoding="utf-8",
    )

    result = release._validate_webview(root)
    assert result["offline_verified"] is True

    auth_path = root / "authenticode.json"
    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    auth["path"] = str(root / "offline" / release.WEBVIEW2_INSTALLER_NAME)
    auth_path.write_text(json.dumps(auth), encoding="utf-8")
    with pytest.raises(release.ReleaseError, match="bound to the accepted installer"):
        release._validate_webview(root)


def test_prepare_writes_manifest_and_rejects_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _write_inputs(tmp_path)
    _stable_source(monkeypatch)
    options = _options(tmp_path, inputs)

    receipt = release.run_release(options)
    assert receipt["package_status"] == "PREPARED_ONLY"
    package = options.dist_root / "TreeLineWarning"
    manifest_path = package / "release_file_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {item["path"]: item for item in manifest["files"]}
    assert "VERSION.txt" in listed
    assert "PACKAGED_ACCEPTANCE_CONTRACT.md" in listed
    assert "release_file_manifest.json" not in listed
    for relative, item in listed.items():
        path = package / relative
        assert path.stat().st_size == item["bytes"]
        assert release._sha256(path) == item["sha256"]
    contract = (package / "PACKAGED_ACCEPTANCE_CONTRACT.md").read_text(encoding="utf-8")
    assert "offline_installer_path: NOT_SELECTED" in contract
    assert "clean_pc: NOT_TESTED" in contract

    snapshot = {
        path.relative_to(options.dist_root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in options.dist_root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(release.ReleaseError, match="refusing to reuse existing"):
        release.run_release(options)
    after = {
        path.relative_to(options.dist_root).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in options.dist_root.rglob("*")
        if path.is_file()
    }
    assert after == snapshot


def test_execute_requires_explicit_overlay_and_webview(tmp_path: Path) -> None:
    options = release.ReleaseOptions(
        dist_root=tmp_path / "dist",
        work_root=tmp_path / "work",
        native_runtime=tmp_path / "native",
        proj_data_root=tmp_path / "proj",
        python_executable=tmp_path / "python.exe",
        spec_path=tmp_path / "spec",
        guide_path=tmp_path / "guide",
        notice_text=tmp_path / "notice.txt",
        notice_json=tmp_path / "notice.json",
        webview_root=tmp_path / "webview",
        execute=True,
        include_webview=False,
    )
    with pytest.raises(release.ReleaseError, match="requires the verified WebView2 installer"):
        release.run_release(options)
    assert not options.dist_root.exists()
    assert not options.work_root.exists()


@pytest.mark.parametrize("has_exe", [False, True])
def test_execute_success_requires_actual_launcher_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_exe: bool
) -> None:
    inputs = _write_inputs(tmp_path)
    _stable_source(monkeypatch)
    monkeypatch.setattr(
        release,
        "_validate_webview",
        lambda root: {
            "source_root": str(root),
            "installer_sha256": "fixture",
            "installer_bytes": 1,
            "offline_verified": True,
            "executed": False,
            "clean_pc": "NOT_TESTED",
        },
    )

    def fake_pyinstaller(options: release.ReleaseOptions, work_root: Path) -> dict[str, object]:
        package = options.dist_root / "TreeLineWarning"
        package.mkdir(parents=True)
        if has_exe:
            (package / "TreeLineWarning.exe").write_bytes(b"exe")
        return {"returncode": 0, "timed_out": False, "error": None}

    monkeypatch.setattr(release, "_run_pyinstaller", fake_pyinstaller)
    monkeypatch.setattr(
        release,
        "_copy_package_metadata",
        lambda package_root, options, **kwargs: {
            "package_root": str(package_root),
            "package_status": "BUILT",
            "webview": kwargs["webview"],
            "file_manifest": str(package_root / "release_file_manifest.json"),
        },
    )
    options = _options(
        tmp_path,
        inputs,
        dist_root=tmp_path / "dist-execute",
        work_root=tmp_path / "work-execute",
        proj_data_root=inputs["proj"],
        execute=True,
        include_webview=True,
    )

    receipt = release.run_release(options)
    assert receipt["package_status"] == ("BUILT" if has_exe else "BUILD_FAILED")
