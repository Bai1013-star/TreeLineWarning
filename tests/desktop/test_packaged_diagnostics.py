from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import desktop.app as app
import desktop.packaged_diagnostics as diagnostics
import pytest


def test_write_json_exclusive_refuses_to_replace_old_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "health.json"

    diagnostics.write_json_exclusive(path, {"run": 1})

    assert json.loads(path.read_text(encoding="utf-8"))["run"] == 1
    with pytest.raises(FileExistsError):
        diagnostics.write_json_exclusive(path, {"run": 2})


def test_dependency_report_contains_runtime_and_module_origins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import desktop.pdal_runtime as runtime

    monkeypatch.setattr(
        runtime,
        "pillow_health",
        lambda: {"found": True, "minimal_operation_pass": True},
    )
    monkeypatch.setattr(
        runtime,
        "dependency_health",
        lambda: {"found": True, "minimal_operation_pass": True},
    )
    monkeypatch.setattr(
        diagnostics,
        "webview2_prerequisites",
        lambda _root: {"ok": True, "status": "PASS"},
    )
    monkeypatch.setattr(
        diagnostics,
        "module_origins",
        lambda: {
            "desktop.app": {"found": True, "origin": "frozen"},
            "lazrs": {"found": True, "origin": "frozen"},
            "webview": {"found": True, "origin": "frozen"},
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "webview_loader_info",
        lambda *_args: {"ok": True, "status": "PASS"},
    )

    report = diagnostics.dependency_report(tmp_path)

    assert report["overall_pass"] is True
    assert report["runtime"]["resource_root"] == str(tmp_path.resolve())
    assert report["module_origins"]["desktop.app"]["origin"] == "frozen"
    assert "lazrs" in report["module_origins"]
    assert report["native_operations"]["pdal"]["minimal_operation_pass"] is True


def test_dependency_report_serializes_native_import_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_runtime = types.ModuleType("desktop.pdal_runtime")
    fake_runtime.__dict__["__loader__"] = None
    monkeypatch.setitem(sys.modules, "desktop.pdal_runtime", fake_runtime)
    monkeypatch.setattr(
        diagnostics,
        "webview2_prerequisites",
        lambda _root: {"ok": True, "status": "PASS"},
    )
    monkeypatch.setattr(
        diagnostics,
        "module_origins",
        lambda: {},
    )

    report = diagnostics.dependency_report(tmp_path)

    assert report["overall_pass"] is False
    assert report["native_operations"]["pillow"]["found"] is False
    assert "cannot import name" in report["native_operations"]["pdal"]["error"]


def test_webview2_check_reports_missing_native_prerequisite_and_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics, "_registry_rows", lambda *_args: [])
    monkeypatch.setattr(diagnostics, "_pywebview_registry_rows", lambda *_args, **_kwargs: [])

    report = diagnostics.webview2_prerequisites(tmp_path)

    assert report["ok"] is False
    assert report["status"] == "MISSING"
    assert "WebView2" in report["message"]
    assert "MSHTML" in report["message"]
    assert report["offline_installer"]["expected_relative_path"] in report["message"]
    with pytest.raises(diagnostics.PackagedPrerequisiteError):
        diagnostics.require_webview2_prerequisites(tmp_path)


def test_webview2_check_accepts_registered_runtime_and_dotnet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")

    def registry_rows(
        subkey: str,
        _names: tuple[str, ...],
        **_kwargs: object,
    ) -> list[dict]:
        if "EdgeUpdate" in subkey:
            return [{"hive": "HKLM", "view": 0, "values": {"pv": "151.0.3482.0"}}]
        return [{"hive": "HKLM", "view": 0, "values": {"Release": 394802}}]

    monkeypatch.setattr(diagnostics, "_registry_rows", registry_rows)
    monkeypatch.setattr(diagnostics, "_pywebview_registry_rows", registry_rows)

    report = diagnostics.webview2_prerequisites(tmp_path)

    assert report["ok"] is True
    assert report["webview2"]["version"] == "151.0.3482.0"
    assert report["dotnet"]["release"] == 394802


def test_webview2_check_rejects_malformed_and_old_registry_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")

    def registry_rows(
        subkey: str,
        _names: tuple[str, ...],
        **_kwargs: object,
    ) -> list[dict]:
        if "EdgeUpdate" in subkey:
            return [
                {"hive": "HKLM", "view": 0, "values": {"pv": "0.0.0.0"}},
                {"hive": "HKCU", "view": 0, "values": {"pv": "85.0.0.0"}},
                {"hive": "HKCU", "view": 0, "values": {"version": "151.0.4129.107"}},
            ]
        return [{"hive": "HKLM", "view": 0, "values": {"Release": 394802}}]

    monkeypatch.setattr(diagnostics, "_registry_rows", registry_rows)
    monkeypatch.setattr(diagnostics, "_pywebview_registry_rows", registry_rows)

    report = diagnostics.webview2_prerequisites(tmp_path)

    assert report["ok"] is False
    assert report["webview2"]["found"] is False
    assert len(report["webview2"]["rejected_versions"]) == 3
    assert report["webview2"]["minimum_version"] == "86.0.622.0"
    assert report["webview2"]["rejected_versions"][-1]["field"] == "pv"


def test_webview2_check_does_not_trust_registry_rows_pywebview_cannot_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    valid_rows = [
        {"hive": "HKLM", "view": 0, "values": {"pv": "151.0.4129.107"}}
    ]
    monkeypatch.setattr(diagnostics, "_registry_rows", lambda *_args: valid_rows)
    monkeypatch.setattr(
        diagnostics,
        "_pywebview_registry_rows",
        lambda *_args, **_kwargs: [],
    )

    report = diagnostics.webview2_prerequisites(tmp_path)

    assert report["ok"] is False
    assert report["webview2"]["found"] is False
    assert report["webview2"]["registry_rows"] == valid_rows
    assert report["webview2"]["pywebview_registry_rows"] == []


def test_pywebview_registry_probe_uses_hklm_wow6432node_on_amd64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, str]] = []

    class Key:
        def __enter__(self) -> Key:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeWinreg:
        HKEY_CURRENT_USER = 1
        HKEY_LOCAL_MACHINE = 2

        @staticmethod
        def OpenKey(hive: int, path: str) -> Key:
            calls.append((hive, path))
            if hive == 2 and "WOW6432Node" in path:
                return Key()
            raise OSError("registry row absent")

        @staticmethod
        def QueryValueEx(_key: Key, _name: str) -> tuple[str, int]:
            return "151.0.4129.107", 1

    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics.platform, "machine", lambda: "AMD64")
    monkeypatch.setitem(sys.modules, "winreg", FakeWinreg)

    rows = diagnostics._pywebview_registry_rows(
        "SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\GUID",
        ("pv",),
    )

    assert rows[0]["hive"] == "HKEY_LOCAL_MACHINE"
    assert "SOFTWARE\\WOW6432Node\\Microsoft\\EdgeUpdate" in rows[0]["registry_path"]
    assert (1, "SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\GUID") in calls
    assert (2, "SOFTWARE\\Microsoft\\EdgeUpdate\\Clients\\GUID") not in calls


def test_offline_installer_reports_size_and_unchecked_hash_separately(
    tmp_path: Path,
) -> None:
    installer = tmp_path / diagnostics.WEBVIEW2_INSTALLER_RELATIVE
    installer.parent.mkdir(parents=True)
    installer.write_bytes(b"test")

    report = diagnostics.offline_installer_info(tmp_path)

    assert report["exists"] is True
    assert report["size_matches"] is False
    assert report["hash_matches"] is None
    assert report["hash_verification"] == "NOT_CHECKED"


def test_offline_installer_finds_package_root_when_resource_root_is_internal(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "TreeLineWarning package"
    internal = package_root / "_internal"
    installer = package_root / diagnostics.WEBVIEW2_INSTALLER_RELATIVE
    installer.parent.mkdir(parents=True)
    installer.write_bytes(b"test")

    report = diagnostics.offline_installer_info(internal)

    assert Path(report["path"]) == installer.resolve()


def test_renderer_preflight_refuses_missing_webview_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(
        diagnostics,
        "webview_loader_info",
        lambda *_args: {
            "ok": False,
            "path": "missing WebView2Loader.dll",
            "error": "WebView2Loader.dll is missing",
        },
    )

    with pytest.raises(diagnostics.PackagedPrerequisiteError, match="MSHTML"):
        diagnostics.require_edgechromium_renderer(object(), tmp_path)


def test_renderer_preflight_refuses_stale_browser_api_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "webview_loader_info",
        lambda *_args: {
            "ok": False,
            "path": "WebView2Loader.dll",
            "error": "installed WebView2 browser is below 86.0.622.0",
            "browser_version": "85.0.0.0",
        },
    )

    with pytest.raises(diagnostics.PackagedPrerequisiteError, match="below 86.0.622.0"):
        diagnostics.require_edgechromium_renderer(object(), tmp_path)


@pytest.mark.parametrize("origin_kind", ("module", "diagnostic_string"))
def test_webview_loader_preflight_reports_pe_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, origin_kind: str
) -> None:
    package_root = tmp_path / "bundle"
    module_file = package_root / "webview" / "__init__.py"
    loader = package_root / diagnostics.WEBVIEW2_LOADER_RELATIVE
    module_file.parent.mkdir(parents=True)
    loader.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")
    loader.write_bytes(b"loader")
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics, "_windows_file_version", lambda _path: "1.0.2957.106")
    monkeypatch.setattr(
        diagnostics,
        "_available_browser_version",
        lambda _path: "151.0.4129.107",
    )

    origin = (
        str(module_file)
        if origin_kind == "diagnostic_string"
        else types.SimpleNamespace(__file__=module_file)
    )
    info = diagnostics.webview_loader_info(tmp_path, origin)

    assert info["ok"] is True
    assert info["loader_file_version"] == "1.0.2957.106"
    assert info["browser_version"] == "151.0.4129.107"
    assert info["path"] == str(loader.resolve())


def test_webview_loader_preflight_rejects_stale_browser_api_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package_root = tmp_path / "bundle"
    module_file = package_root / "webview" / "__init__.py"
    loader = package_root / diagnostics.WEBVIEW2_LOADER_RELATIVE
    module_file.parent.mkdir(parents=True)
    loader.parent.mkdir(parents=True)
    module_file.write_text("", encoding="utf-8")
    loader.write_bytes(b"loader")
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics, "_windows_file_version", lambda _path: "1.0.2957.106")
    monkeypatch.setattr(diagnostics, "_available_browser_version", lambda _path: "85.0.0.0")

    info = diagnostics.webview_loader_info(tmp_path, types.SimpleNamespace(__file__=module_file))

    assert info["ok"] is False
    assert info["browser_version_valid"] is False
    assert "below 86.0.622.0" in info["error"]


def test_current_survey_smoke_is_read_only_and_closes_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "stage-a-project.json").write_text("{}", encoding="utf-8")
    state = {"stopped": False}

    class FakeBackend:
        def __init__(self, project_root: Path, repository_root: Path) -> None:
            state["project_root"] = project_root
            state["repository_root"] = repository_root

        def start(self) -> str:
            return "http://127.0.0.1:4567/"

        def stop(self) -> None:
            state["stopped"] = True

    payload = {
        "ok": True,
        "schema": "treeline-current-final-report-v1",
        "current_only": True,
        "status": "FINALIZED",
        "gate": {"status": "PASS"},
        "kpis": {"tree_count": 1},
        "trees": [{"persistent_tree_id": "T1"}],
        "wires": {
            "source": "PERSISTED_SPAN_WIRE_RESULT_W1",
            "geometry_kind": "FITTED_CURRENT_GEOMETRY",
            "segments": [{"span_id": "S1", "xyz": [[0, 0, 0], [1, 0, 1]]}],
        },
    }
    monkeypatch.setattr(app, "BackendProcess", FakeBackend)
    monkeypatch.setattr(diagnostics, "_read_json", lambda *_args, **_kwargs: payload)

    report = diagnostics.current_survey_smoke(tmp_path, tmp_path)

    assert report["status"] == "PASS"
    assert report["checks"]["w1_curves_are_persisted"] is True
    assert report["read_only_actions"]["prediction_export_requested"] is False
    assert state["stopped"] is True


def test_finite_backend_smoke_publishes_url_and_stops_on_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "desktop-project.json").write_text("{}", encoding="utf-8")
    state = {"stopped": False}

    class FakeBackend:
        def __init__(self, *_args) -> None:
            pass

        def start(self) -> str:
            return "http://127.0.0.1:9876/"

        def stop(self) -> None:
            state["stopped"] = True

    monkeypatch.setattr(app, "BackendProcess", FakeBackend)
    output = tmp_path / "backend.json"
    stop = tmp_path / "stop.signal"
    stop.write_text("stop", encoding="utf-8")

    code, result = diagnostics.finite_backend_smoke(
        tmp_path,
        tmp_path,
        output,
        stop_signal=stop,
        timeout_seconds=1,
    )

    completion = tmp_path / "backend.complete.json"
    assert code == 0
    assert result["status"] == "STOPPED"
    assert json.loads(output.read_text(encoding="utf-8"))["backend_url"].endswith("/")
    assert json.loads(completion.read_text(encoding="utf-8"))["status"] == "STOPPED"
    assert state["stopped"] is True


def test_finite_backend_smoke_rejects_unbounded_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at most 600"):
        diagnostics.finite_backend_smoke(
            tmp_path,
            tmp_path,
            tmp_path / "backend.json",
            timeout_seconds=diagnostics.MAX_SMOKE_TIMEOUT_SECONDS + 1,
        )


def test_dependency_health_cli_writes_new_path_and_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "resource_root", lambda: tmp_path)
    monkeypatch.setattr(
        app,
        "dependency_report",
        lambda _root: {"schema": "test", "overall_pass": True},
    )
    output = tmp_path / "health.json"

    assert app.main(["--dependency-health", "--diagnostic-output", str(output)]) == 0
    assert app.main(["--dependency-health", "--diagnostic-output", str(output)]) == 2


def test_dependency_health_cli_failure_writes_nonzero_diagnostic_without_dialog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "resource_root", lambda: tmp_path)

    def fail_report(_root: Path) -> dict:
        raise RuntimeError("dependency probe failed")

    monkeypatch.setattr(
        app,
        "dependency_report",
        fail_report,
    )
    monkeypatch.setattr(
        app,
        "_show_gui_error",
        lambda _message: pytest.fail("dependency-health CLI must not open a GUI error dialog"),
    )
    output = tmp_path / "dependency-failure.json"

    assert app.main(["--dependency-health", "--diagnostic-output", str(output)]) == 5
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["mode"] == "dependency-health"
    assert payload["error"] == "dependency probe failed"


def test_launch_reports_missing_prerequisite_to_file_before_importing_webview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app, "resource_root", lambda: tmp_path)
    error = diagnostics.PackagedPrerequisiteError("missing WebView2; use offline installer")

    def fail_prerequisite(_root: Path) -> dict:
        raise error

    monkeypatch.setattr(app, "require_webview2_prerequisites", fail_prerequisite)
    shown: list[str] = []
    monkeypatch.setattr(app, "_show_gui_error", shown.append)
    output = tmp_path / "launch.json"

    assert app.launch(None, diagnostic_output=output) == 3
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "MISSING_PREREQUISITE"
    assert "offline installer" in payload["error"]
    assert shown == [str(error)]


def test_gui_launch_records_lifecycle_and_unverified_visual_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stage-a-project.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "launch-success.json"
    state = {"stopped": False}

    class FakeBackend:
        def __init__(self, project_root: Path, *_args: object, **_kwargs: object) -> None:
            self.project_root = project_root

        def start(self) -> str:
            return "http://127.0.0.1:4567/"

        def stop(self) -> None:
            state["stopped"] = True

    fake_webview = types.ModuleType("webview")
    fake_webview.renderer = "edgechromium"
    fake_webview.create_window = lambda *_args, **_kwargs: object()
    fake_webview.start = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(app, "resource_root", lambda: tmp_path)
    monkeypatch.setattr(app, "BackendProcess", FakeBackend)
    monkeypatch.setattr(app, "require_webview2_prerequisites", lambda _root: {"ok": True})
    monkeypatch.setattr(
        app,
        "require_edgechromium_renderer",
        lambda *_args: {
            "requested_renderer": "edgechromium",
            "webview_loader": {"ok": True, "version": "1.0.2957.106"},
        },
    )

    assert app.launch(project, diagnostic_output=output) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["renderer"] == "edgechromium"
    assert payload["lifecycle"] == "WINDOW_OPENED_AND_CLOSED"
    assert payload["visual_acceptance"] == "UNVERIFIED"
    assert state["stopped"] is True


def test_gui_launch_rejects_renderer_fallback_after_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "stage-a-project.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "renderer-failure.json"
    state = {"stopped": False}

    class FakeBackend:
        def __init__(self, project_root: Path, *_args: object, **_kwargs: object) -> None:
            self.project_root = project_root

        def start(self) -> str:
            return "http://127.0.0.1:4567/"

        def stop(self) -> None:
            state["stopped"] = True

    fake_webview = types.ModuleType("webview")
    fake_webview.renderer = "mshtml"
    fake_webview.create_window = lambda *_args, **_kwargs: object()
    fake_webview.start = lambda **_kwargs: None
    monkeypatch.setitem(sys.modules, "webview", fake_webview)
    monkeypatch.setattr(app, "resource_root", lambda: tmp_path)
    monkeypatch.setattr(app, "BackendProcess", FakeBackend)
    monkeypatch.setattr(app, "_show_gui_error", lambda _message: None)
    monkeypatch.setattr(app, "require_webview2_prerequisites", lambda _root: {"ok": True})
    monkeypatch.setattr(
        app,
        "require_edgechromium_renderer",
        lambda *_args: {"requested_renderer": "edgechromium", "webview_loader": {"ok": True}},
    )

    assert app.launch(project, diagnostic_output=output) == 3
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "MISSING_PREREQUISITE"
    assert "MSHTML" in payload["error"]
    assert state["stopped"] is True


def test_cli_survey_failure_returns_nonzero_and_writes_failure_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        app,
        "_show_gui_error",
        lambda _message: pytest.fail("CLI smoke must not open a GUI error dialog"),
    )
    output = tmp_path / "survey-failure.json"

    assert app.main(["--survey-smoke", "--diagnostic-output", str(output)]) == 5
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "FAIL"


def test_cli_backend_failure_returns_nonzero_and_writes_failure_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "backend-failure.json"

    def fail_backend(*_args: object, **_kwargs: object) -> tuple[int, dict]:
        raise RuntimeError("backend probe failed")

    monkeypatch.setattr(app, "finite_backend_smoke", fail_backend)

    assert app.main(
        [
            "--backend-smoke",
            "--project-root",
            str(tmp_path),
            "--diagnostic-output",
            str(output),
        ]
    ) == 5
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "backend probe failed" in payload["error"]


def test_native_gui_error_uses_message_box_without_showing_one_in_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, str, str, int]] = []
    fake_ctypes = types.ModuleType("ctypes")
    fake_ctypes.windll = types.SimpleNamespace(
        user32=types.SimpleNamespace(
            MessageBoxW=lambda owner, message, title, flags: calls.append(
                (owner, message, title, flags)
            )
        )
    )
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)
    monkeypatch.setattr(app.sys, "platform", "win32")

    app._show_gui_error("native failure")

    assert calls == [(None, "native failure", "TreeLineWarning", 0x10)]


def test_frozen_entry_calls_freeze_support() -> None:
    source = Path(app.__file__).read_text(encoding="utf-8")
    assert "multiprocessing.freeze_support()" in source
