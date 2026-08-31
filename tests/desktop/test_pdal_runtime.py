from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
from desktop.pdal_runtime import dependency_health

from desktop import pdal_runtime


@pytest.fixture(autouse=True)
def isolate_native_configuration(monkeypatch):
    monkeypatch.setattr(pdal_runtime, "_NATIVE_RUNTIME_PAIR", None)
    monkeypatch.setattr(pdal_runtime, "_NATIVE_PROJ_CONFIGURED", False)
    monkeypatch.setattr(pdal_runtime, "_DLL_HANDLES", [])


@pytest.fixture
def native_api(tmp_path, monkeypatch):
    runtime = tmp_path / "native runtime"
    runtime.mkdir()
    for name in ("proj_9.dll", "gdal.dll"):
        (runtime / name).write_bytes(b"mock DLL")
    data = tmp_path / "中文 数据"
    data.mkdir()
    (data / "proj.db").write_bytes(b"mock database")
    calls = []

    def proj_paths(context, count, paths):
        calls.append(("proj_paths", context, count, paths[0]))

    def proj_network(context, enabled):
        calls.append(("proj_network", context, enabled))
        return enabled

    def gdal_paths(paths):
        calls.append(("gdal_paths", paths[0], paths[1]))

    def gdal_network(enabled):
        calls.append(("gdal_network", enabled))

    proj = SimpleNamespace(
        proj_context_set_search_paths=proj_paths,
        proj_context_set_enable_network=proj_network,
    )
    gdal = SimpleNamespace(
        OSRSetPROJSearchPaths=gdal_paths,
        OSRSetPROJEnableNetwork=gdal_network,
    )
    libraries = {str(runtime / "proj_9.dll"): proj, str(runtime / "gdal.dll"): gdal}

    def load(path):
        calls.append(("load", path))
        return libraries[path]

    monkeypatch.setattr(pdal_runtime.ctypes, "CDLL", load)
    return runtime, data, proj, gdal, calls


def test_pdal_health_probe_runs_the_real_minimal_operation_when_available() -> None:
    pytest.importorskip("pdal")
    report = dependency_health()
    assert report["found"] is True
    assert report["minimal_operation_pass"] is True


def test_frozen_runtime_does_not_use_external_native_overrides(tmp_path, monkeypatch) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    external = tmp_path / "external"
    (external / "Library/bin").mkdir(parents=True)
    data = external / "Library/share/proj"
    data.mkdir(parents=True)
    (data / "proj.db").write_bytes(b"external database")
    monkeypatch.setattr(pdal_runtime.sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(pdal_runtime.sys, "frozen", True, raising=False)
    monkeypatch.setenv("TREELINE_PDAL_RUNTIME_ENV", str(external))
    monkeypatch.setenv("TREELINE_PDAL_PROJ_DATA_ROOT", str(external))
    monkeypatch.setenv("PROJ_NETWORK", "OFF")

    assert pdal_runtime.bundled_proj_data_dir() is None
    assert pdal_runtime._configure_bundled_runtime() == (None, [])


@pytest.mark.parametrize("library, label", [("proj_9.dll", "PROJ"), ("gdal.dll", "GDAL")])
def test_native_proj_configuration_fails_closed_without_library(
    native_api, library, label
) -> None:
    runtime, data, _proj, _gdal, calls = native_api
    (runtime / library).unlink()

    with pytest.raises(RuntimeError, match=f"native {label} library is unavailable"):
        pdal_runtime._configure_native_proj_search_path(data, runtime)
    assert not calls
    assert pdal_runtime._NATIVE_PROJ_CONFIGURED is False


@pytest.mark.parametrize(
    "label, api", [
        ("PROJ", "proj_context_set_search_paths"),
        ("PROJ", "proj_context_set_enable_network"),
        ("GDAL", "OSRSetPROJSearchPaths"),
        ("GDAL", "OSRSetPROJEnableNetwork"),
    ]
)
def test_native_proj_configuration_fails_closed_without_required_api(
    native_api, label, api
) -> None:
    runtime, data, proj, gdal, calls = native_api
    delattr(proj if label == "PROJ" else gdal, api)
    message = f"native {label} library does not expose the required API"
    with pytest.raises(RuntimeError, match=message):
        pdal_runtime._configure_native_proj_search_path(data, runtime)
    assert all(call[0] == "load" for call in calls)
    assert pdal_runtime._NATIVE_PROJ_CONFIGURED is False
    with pytest.raises(RuntimeError, match="cannot be switched"):
        pdal_runtime._configure_native_proj_search_path(data / "other", runtime)


def test_native_proj_configuration_fails_closed_without_database(native_api) -> None:
    runtime, data, _proj, _gdal, calls = native_api
    (data / "proj.db").unlink()
    with pytest.raises(RuntimeError, match="native PROJ data is unavailable"):
        pdal_runtime._configure_native_proj_search_path(data, runtime)
    assert not calls
    assert pdal_runtime._NATIVE_PROJ_CONFIGURED is False


@pytest.mark.parametrize("library", ["proj_9.dll", "gdal.dll"])
def test_native_proj_configuration_fails_closed_on_dll_load_error(
    native_api, monkeypatch, library
) -> None:
    runtime, data, _proj, _gdal, _calls = native_api
    original_loader = pdal_runtime.ctypes.CDLL

    def load(path):
        if path == str(runtime / library):
            raise OSError("native dependency cannot be loaded")
        return original_loader(path)

    monkeypatch.setattr(pdal_runtime.ctypes, "CDLL", load)
    with pytest.raises(RuntimeError, match="does not expose the required API"):
        pdal_runtime._configure_native_proj_search_path(data, runtime)
    assert pdal_runtime._NATIVE_PROJ_CONFIGURED is False


def test_native_proj_uses_exact_dlls_utf8_paths_and_correct_api_signatures(native_api) -> None:
    runtime, data, proj, gdal, calls = native_api
    pdal_runtime._configure_native_proj_search_path(data, runtime)
    assert calls == [
        ("load", str(runtime / "proj_9.dll")),
        ("load", str(runtime / "gdal.dll")),
        ("proj_paths", None, 1, str(data).encode("utf-8")),
        ("proj_network", None, 0),
        ("gdal_paths", str(data).encode("utf-8"), None),
        ("gdal_network", 0),
    ]
    assert proj.proj_context_set_search_paths.argtypes == [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)
    ]
    assert proj.proj_context_set_search_paths.restype is None
    assert proj.proj_context_set_enable_network.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert proj.proj_context_set_enable_network.restype is ctypes.c_int
    assert gdal.OSRSetPROJSearchPaths.argtypes == [ctypes.POINTER(ctypes.c_char_p)]
    assert gdal.OSRSetPROJSearchPaths.restype is None
    assert gdal.OSRSetPROJEnableNetwork.argtypes == [ctypes.c_int]
    assert gdal.OSRSetPROJEnableNetwork.restype is None
    assert pdal_runtime._DLL_HANDLES == [proj, gdal]


@pytest.mark.parametrize("change", ["runtime", "data"])
def test_native_runtime_rejects_switch_before_side_effects_and_reuses_original_pair(
    native_api, monkeypatch, change
) -> None:
    runtime, data, _proj, _gdal, calls = native_api
    monkeypatch.setattr(pdal_runtime, "bundled_runtime_dir", lambda: runtime)
    monkeypatch.setattr(pdal_runtime, "bundled_proj_data_dir", lambda: data)
    monkeypatch.setattr(pdal_runtime.os, "environ", dict(pdal_runtime.os.environ))
    monkeypatch.setattr(
        pdal_runtime, "_preload_native_sqlite", lambda path: calls.append(("sqlite", path))
    )
    monkeypatch.setattr(
        pdal_runtime.os, "add_dll_directory", lambda path: object(), raising=False
    )
    first_result = pdal_runtime._configure_bundled_runtime()
    first_calls = list(calls)
    first_environment = dict(pdal_runtime.os.environ)
    other = runtime.parent / "other"
    other.mkdir()
    function = "bundled_runtime_dir" if change == "runtime" else "bundled_proj_data_dir"
    original = runtime if change == "runtime" else data
    monkeypatch.setattr(pdal_runtime, function, lambda: other)
    with pytest.raises(RuntimeError, match="cannot be switched"):
        pdal_runtime._configure_bundled_runtime()
    assert pdal_runtime.os.environ == first_environment
    assert calls == first_calls
    monkeypatch.setattr(pdal_runtime, function, lambda: original)
    assert pdal_runtime._configure_bundled_runtime() == first_result
    assert calls == first_calls
    assert pdal_runtime.os.environ == first_environment
