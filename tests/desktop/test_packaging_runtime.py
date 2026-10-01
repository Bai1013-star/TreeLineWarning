from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import laspy
import numpy as np
import pytest
from scripts.prepare_desktop_native_runtime import _is_reparse_point

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path(os.environ.get(
    "TREELINE_TEST_PDAL_STAGE",
    REPO_ROOT / ".build-pdal-runtime-final-20260830",
))
PROJ_DATA_ROOT = Path(os.environ.get(
    "TREELINE_TEST_PDAL_PROJ_DATA_ROOT",
    REPO_ROOT / ".build-pdal-proj-data-final-20260830-02",
))


def _write_small_las(path: Path) -> laspy.LasData:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([0.0, 1.25, 2.5, 3.75, 4.0])
    cloud.y = np.asarray([0.0, 2.5, 1.25, 3.0, 4.5])
    cloud.z = np.asarray([10.0, 10.25, 11.5, 12.0, 13.75])
    cloud.intensity = np.asarray([10, 20, 30, 40, 50], dtype=np.uint16)
    cloud.classification = np.asarray([2, 5, 5, 2, 1], dtype=np.uint8)
    cloud.write(path)
    return cloud


def _write_probe_las(path: Path) -> None:
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 19.0, 20), np.linspace(0.0, 19.0, 20))
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    cloud = laspy.LasData(header)
    x = grid_x.ravel()
    y = grid_y.ravel()
    cloud.x = x
    cloud.y = y
    cloud.z = 1.0 + 0.01 * x + 0.01 * y
    cloud.classification = np.zeros(len(cloud.points), dtype=np.uint8)
    cloud.write(path)


def _write_epsg_las(path: Path) -> None:
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.asarray([0.01, 0.01, 0.01])
    cloud = laspy.LasData(header)
    cloud.x = np.asarray([120.0])
    cloud.y = np.asarray([30.0])
    cloud.z = np.asarray([1.0])
    cloud.intensity = np.asarray([1], dtype=np.uint16)
    cloud.classification = np.asarray([0], dtype=np.uint8)
    cloud.write(path)


def _require_staged_runtime() -> None:
    site = RUNTIME_ROOT / "Lib" / "site-packages"
    runtime = RUNTIME_ROOT / "Library" / "bin"
    proj_data = PROJ_DATA_ROOT / "Library" / "share" / "proj"
    if os.name != "nt" or not (
        site.is_dir()
        and all((runtime / name).is_file() for name in (
            "pdalcpp.dll", "proj_9.dll", "gdal.dll", "sqlite3.dll"
        ))
        and (proj_data / "proj.db").is_file()
    ):
        pytest.skip("materialized PDAL/PROJ runtime is not present")


def _run_staged_probe(code: str, *arguments: Path) -> subprocess.CompletedProcess[str]:
    _require_staged_runtime()
    site = RUNTIME_ROOT / "Lib" / "site-packages"
    runtime = RUNTIME_ROOT / "Library" / "bin"
    proj_data = PROJ_DATA_ROOT / "Library" / "share" / "proj"
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join([str(runtime), environment.get("PATH", "")])
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(site), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["PDAL_DRIVER_PATH"] = str(runtime)
    environment["TREELINE_TEST_PDAL_STAGE"] = str(RUNTIME_ROOT)
    environment["TREELINE_PDAL_RUNTIME_ENV"] = str(RUNTIME_ROOT)
    environment["TREELINE_TEST_PDAL_PROJ_DATA"] = str(proj_data)
    environment["TREELINE_PDAL_PROJ_DATA_ROOT"] = str(PROJ_DATA_ROOT)
    environment["PROJ_DATA"] = str(proj_data)
    probe_python = REPO_ROOT / ".venv311" / "Scripts" / "python.exe"
    if not probe_python.is_file():
        probe_python = Path(sys.executable)
    return subprocess.run(
        [str(probe_python), "-c", code, *(str(path) for path in arguments)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_materialized_runtime_manifest_covers_real_required_payload() -> None:
    if not RUNTIME_ROOT.is_dir():
        pytest.skip("materialized PDAL runtime is not present")

    manifest_path = RUNTIME_ROOT / "native_runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["versions"]["pdal_python_binding"] == "3.5.4"
    assert manifest["versions"]["pdal"] == "2.10.2"
    assert manifest["versions"]["gdal"] == "3.13.1"
    assert manifest["versions"]["proj"] == "9.8.1"
    assert manifest["compatibility_files"]
    assert {row["destination_relative"] for row in manifest["compatibility_files"]} == {
        "compat/openjp2.dll",
        "compat/zlib-ng2.dll",
    }

    rows = [*manifest["files"], *manifest["compatibility_files"]]
    assert rows
    assert all(row.get("machine") == "AMD64" for row in rows if row.get("format") == "PE")
    for row in rows:
        path = RUNTIME_ROOT / Path(row["destination_relative"])
        assert path.is_file()
        assert not _is_reparse_point(path)
        assert path.stat().st_size == row["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]

    assert not any(_is_reparse_point(path) for path in RUNTIME_ROOT.rglob("*"))
    assert (RUNTIME_ROOT / "Lib/site-packages/pdal/libpdalpython.cp311-win_amd64.pyd").is_file()
    assert (RUNTIME_ROOT / "Library/bin/pdalcpp.dll").is_file()
    assert not (RUNTIME_ROOT / "pdal_plugins").exists()
    assert not list((RUNTIME_ROOT / "Lib/site-packages/bin").glob("*.dll"))


def test_release_viewer_payload_contains_current_report_assets() -> None:
    viewer_root = REPO_ROOT / "viewer_poc" / "giro3d"
    for name in ("current_report.html", "current_report.css", "current_report.js"):
        assert (viewer_root / name).is_file()


def test_las_laz_round_trip_preserves_point_values(tmp_path: Path) -> None:
    pytest.importorskip("lazrs")
    source_path = tmp_path / "source.las"
    laz_path = tmp_path / "source.laz"
    source = _write_small_las(source_path)
    source.write(laz_path)

    decoded = laspy.read(laz_path)
    np.testing.assert_array_equal(decoded.X, source.X)
    np.testing.assert_array_equal(decoded.Y, source.Y)
    np.testing.assert_array_equal(decoded.Z, source.Z)
    np.testing.assert_array_equal(decoded.intensity, source.intensity)
    np.testing.assert_array_equal(decoded.classification, source.classification)


def test_materialized_native_pdal_reads_laz(tmp_path: Path) -> None:
    pytest.importorskip("lazrs")
    source_path = tmp_path / "source.las"
    laz_path = tmp_path / "source.laz"
    output_path = tmp_path / "pdal-output.las"
    source = _write_small_las(source_path)
    source.write(laz_path)

    result = _run_staged_probe(
        """
import json
import os
import sys
from pathlib import Path

stage = Path(os.environ["TREELINE_TEST_PDAL_STAGE"])
site = stage / "Lib" / "site-packages"
runtime = stage / "Library" / "bin"
sys.path.insert(0, str(site))
os.environ["PATH"] = os.pathsep.join([str(runtime), os.environ.get("PATH", "")])
dll_handle = os.add_dll_directory(str(runtime))
import pdal

module_path = Path(pdal.__file__).resolve()
assert module_path.is_relative_to(site / "pdal"), module_path
pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": sys.argv[1]},
        {"type": "writers.las", "filename": sys.argv[2], "forward": "all"},
    ]
}
processed = int(pdal.Pipeline(json.dumps(pipeline)).execute())
assert processed > 0, processed
dll_handle.close()
""",
        laz_path,
        output_path,
    )
    assert result.returncode == 0, (result.stderr, result.stdout)
    decoded = laspy.read(output_path)
    assert len(decoded.points) == len(source.points)
    np.testing.assert_array_equal(decoded.X, source.X)
    np.testing.assert_array_equal(decoded.Y, source.Y)
    np.testing.assert_array_equal(decoded.Z, source.Z)


def test_staged_runtime_bootstraps_native_sqlite_before_python_and_pdal(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "epsg-input.las"
    output_path = tmp_path / "epsg-output.las"
    _write_epsg_las(input_path)
    result = _run_staged_probe(
        """
import json
import os
import sys
from pathlib import Path

stage = Path(os.environ["TREELINE_TEST_PDAL_STAGE"])
site = stage / "Lib" / "site-packages"
sys.path.insert(0, str(site))
from desktop.pdal_runtime import _configure_bundled_runtime

_configure_bundled_runtime()
import sqlite3

assert sqlite3.sqlite_version == "3.53.3", sqlite3.sqlite_version
connection = sqlite3.connect(":memory:")
try:
    connection.execute("CREATE TABLE probe (value INTEGER)")
    connection.execute("INSERT INTO probe VALUES (1)")
    assert connection.execute("SELECT value FROM probe").fetchone() == (1,)
finally:
    connection.close()

import laspy
import pdal

module_path = Path(pdal.__file__).resolve()
assert module_path.is_relative_to(site / "pdal"), module_path
pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": os.sys.argv[1]},
        {"type": "filters.reprojection", "in_srs": "EPSG:4326", "out_srs": "EPSG:3857"},
        {"type": "writers.las", "filename": os.sys.argv[2]},
    ]
}
assert int(pdal.Pipeline(json.dumps(pipeline)).execute()) == 1
point = laspy.read(os.sys.argv[2])
assert abs(float(point.x[0]) - 13358338.8952) < 0.05, point.x[0]
assert abs(float(point.y[0]) - 3503549.8435) < 0.05, point.y[0]
print(json.dumps({
    "sqlite_version": sqlite3.sqlite_version,
    "pdal_module": str(module_path),
    "pdal_version": str(pdal.info.version),
}, sort_keys=True), flush=True)
""",
        input_path,
        output_path,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["sqlite_version"] == "3.53.3"
    assert payload["pdal_version"] == "2.10.2"


@pytest.mark.parametrize("layout", ["staged", "frozen"])
def test_staged_runtime_configures_unicode_proj_for_native_and_threaded_pdal(
    tmp_path: Path, layout: str,
) -> None:
    _require_staged_runtime()
    unicode_root = tmp_path / "中文 目录"
    native_proj_data = (
        unicode_root / "pdal_proj_data"
        if layout == "frozen"
        else unicode_root / "Library" / "share" / "proj"
    )
    shutil.copytree(PROJ_DATA_ROOT / "Library" / "share" / "proj", native_proj_data)
    if layout == "frozen":
        shutil.copytree(RUNTIME_ROOT / "Library" / "bin", unicode_root / "pdal_runtime")
    io_root = unicode_root / "输入 输出"
    io_root.mkdir()
    input_path = io_root / "输入.las"
    main_output_path = io_root / "主线程输出.las"
    worker_output_path = io_root / "工作线程输出.las"
    _write_epsg_las(input_path)

    result = _run_staged_probe(
        """
import concurrent.futures
import ctypes
import hashlib
import json
import os
import sys
from pathlib import Path

stage = Path(os.environ["TREELINE_TEST_PDAL_STAGE"])
site = stage / "Lib" / "site-packages"
runtime = stage / "Library" / "bin"
unicode_root = Path(sys.argv[3]).resolve()
native_proj_data = unicode_root / "Library" / "share" / "proj"
if sys.argv[5] == "frozen":
    runtime = unicode_root / "pdal_runtime"
    native_proj_data = unicode_root / "pdal_proj_data"
    sys._MEIPASS = str(unicode_root)
    sys.frozen = True
os.environ["TREELINE_PDAL_PROJ_DATA_ROOT"] = str(unicode_root)
os.environ["PROJ_DATA"] = str(native_proj_data)
os.environ["PROJ_NETWORK"] = "OFF"
sys.path.insert(0, str(site))
import pyproj

pyproj_data_before = Path(pyproj.datadir.get_data_dir()).resolve()
pyproj_db_hash_before = hashlib.sha256((pyproj_data_before / "proj.db").read_bytes()).hexdigest()
from desktop import pdal_runtime

configured_runtime, _ = pdal_runtime._configure_bundled_runtime()
assert configured_runtime == runtime
assert os.environ["PROJ_DATA"] == str(native_proj_data)
get_environment = ctypes.WinDLL("kernel32", use_last_error=True).GetEnvironmentVariableW
get_environment.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
get_environment.restype = ctypes.c_uint32
environment_buffer = ctypes.create_unicode_buffer(32768)
assert get_environment("PROJ_DATA", environment_buffer, len(environment_buffer)) > 0
assert environment_buffer.value == str(native_proj_data)
configured_again, _ = pdal_runtime._configure_bundled_runtime()
assert configured_again == runtime

import laspy
import pdal

pyproj.network.set_network_enabled(False)
pyproj_data = Path(pyproj.datadir.get_data_dir()).resolve()
assert pyproj_data != native_proj_data
assert pyproj_data == pyproj_data_before
assert (pyproj_data / "proj.db").is_file(), pyproj_data
transformer = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True, allow_ballpark=False, only_best=True
)
pyproj_x, pyproj_y = transformer.transform(120.0, 30.0, errcheck=True)
assert abs(pyproj_x - 13358338.8952) < 0.05
assert abs(pyproj_y - 3503549.8435) < 0.05

def run_pipeline(output_path, input_path=None):
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(input_path or sys.argv[1])},
            {"type": "filters.reprojection", "in_srs": "EPSG:4326", "out_srs": "EPSG:3857"},
            {"type": "writers.las", "filename": str(output_path)},
        ]
    }
    return int(pdal.Pipeline(json.dumps(pipeline)).execute())

main_processed = run_pipeline(Path(sys.argv[2]))
with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
    worker_processed = executor.submit(run_pipeline, Path(sys.argv[4])).result()
assert main_processed == worker_processed == 1
output_paths = [Path(sys.argv[2]), Path(sys.argv[4])]
laz_processed = None
if laspy.LazBackend.detect_available():
    laz_path = Path(sys.argv[1]).with_suffix(".laz")
    laspy.read(sys.argv[1]).write(laz_path)
    laz_output = Path(sys.argv[4]).with_suffix(".laz")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        laz_processed = executor.submit(run_pipeline, laz_output, laz_path).result()
    assert laz_processed == 1
    output_paths.append(laz_output)
for output_path in output_paths:
    point = laspy.read(output_path)
    assert len(point.points) == 1
    assert abs(float(point.x[0]) - 13358338.8952) < 0.05
    assert abs(float(point.y[0]) - 3503549.8435) < 0.05
proj = ctypes.CDLL(str(runtime / "proj_9.dll"))
gdal = ctypes.CDLL(str(runtime / "gdal.dll"))
proj.proj_context_is_network_enabled.argtypes = [ctypes.c_void_p]
proj.proj_context_is_network_enabled.restype = ctypes.c_int
gdal.OSRGetPROJEnableNetwork.argtypes = []
gdal.OSRGetPROJEnableNetwork.restype = ctypes.c_int
assert proj.proj_context_is_network_enabled(None) == 0
assert gdal.OSRGetPROJEnableNetwork() == 0
assert pyproj.network.is_network_enabled() is False
assert Path(pyproj.datadir.get_data_dir()).resolve() == pyproj_data_before
assert hashlib.sha256((pyproj_data / "proj.db").read_bytes()).hexdigest() == pyproj_db_hash_before
get_module_filename = ctypes.WinDLL("kernel32", use_last_error=True).GetModuleFileNameW
get_module_filename.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
get_module_filename.restype = ctypes.c_uint32
module_paths = []
for library in (proj, gdal, ctypes.WinDLL(str(runtime / "sqlite3.dll"))):
    buffer = ctypes.create_unicode_buffer(32768)
    assert get_module_filename(library._handle, buffer, len(buffer)) > 0
    actual_path = Path(buffer.value).resolve()
    assert actual_path.parent == runtime.resolve(), actual_path
    module_paths.append(str(actual_path))
print(json.dumps({
    "native_proj_data": str(native_proj_data),
    "pyproj_data": str(pyproj_data),
    "main_processed": main_processed,
    "worker_processed": worker_processed,
    "laz_processed": laz_processed,
    "pyproj_database_unchanged": True,
    "wide_environment_preserved": True,
    "native_modules": module_paths,
    "configured": pdal_runtime._NATIVE_PROJ_CONFIGURED,
}, sort_keys=True), flush=True)
""",
        input_path,
        main_output_path,
        unicode_root,
        worker_output_path,
        Path(layout),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["main_processed"] == payload["worker_processed"] == 1
    assert Path(payload["native_proj_data"]) == native_proj_data
    assert payload["pyproj_data"] != payload["native_proj_data"]
    assert payload["configured"] is True


def test_unicode_runtime_skips_before_copying_missing_native_fixtures(tmp_path, monkeypatch):
    monkeypatch.setitem(globals(), "RUNTIME_ROOT", tmp_path / "missing native")

    def unexpected_copy(*_args, **_kwargs):
        raise AssertionError("native fixture must be checked before copying")

    monkeypatch.setattr(shutil, "copytree", unexpected_copy)
    with pytest.raises(pytest.skip.Exception, match="runtime is not present"):
        test_staged_runtime_configures_unicode_proj_for_native_and_threaded_pdal(
            tmp_path, "frozen"
        )


def test_materialized_native_core_and_offline_proj_health(tmp_path: Path) -> None:
    input_path = tmp_path / "probe.las"
    output_path = tmp_path / "probe-output.las"
    _write_probe_las(input_path)

    epsg_result = _run_staged_probe(
        """
import json
import os
import sys
from pathlib import Path

import pyproj

stage = Path(os.environ["TREELINE_TEST_PDAL_STAGE"])
site = stage / "Lib" / "site-packages"
runtime = stage / "Library" / "bin"
sys.path.insert(0, str(site))
os.environ["PATH"] = os.pathsep.join([str(runtime), os.environ.get("PATH", "")])
os.environ["PROJ_NETWORK"] = "OFF"
pyproj.network.set_network_enabled(False)
pyproj_data = Path(pyproj.datadir.get_data_dir()).resolve()
assert (pyproj_data / "proj.db").is_file(), pyproj_data
native_proj_data = Path(os.environ["TREELINE_TEST_PDAL_PROJ_DATA"])
assert (native_proj_data / "proj.db").is_file(), native_proj_data
os.environ["PROJ_DATA"] = str(native_proj_data)
dll_handle = os.add_dll_directory(str(runtime))
import pdal

module_path = Path(pdal.__file__).resolve()
assert module_path.is_relative_to(site / "pdal"), module_path
pdal_info = getattr(pdal, "info", None)
assert pdal_info is not None
native_epsg_error = None
try:
    pdal.Pipeline(json.dumps({
        "pipeline": [
            {"type": "readers.las", "filename": sys.argv[1]},
            {
                "type": "filters.reprojection",
                "in_srs": "EPSG:4326",
                "out_srs": "EPSG:3857",
            },
            {"type": "writers.las", "filename": sys.argv[2]},
        ]
    })).execute()
except RuntimeError as exc:
    native_epsg_error = str(exc)
if native_epsg_error is not None:
    assert "DATABASE.LAYOUT" in native_epsg_error, native_epsg_error
print(json.dumps({
    "pdal_module": str(module_path),
    "pdal_version": str(getattr(pdal_info, "version", "unknown")),
    "proj_data": str(pyproj_data),
    "native_epsg_error": native_epsg_error,
}, sort_keys=True), flush=True)
dll_handle.close()
""",
        input_path,
        output_path.with_name("probe-epsg-output.las"),
    )
    assert epsg_result.returncode == 0, epsg_result.stderr
    epsg_probe = json.loads(epsg_result.stdout.strip().splitlines()[-1])
    assert epsg_probe["pdal_version"] == "2.10.2"
    native_epsg_error = epsg_probe["native_epsg_error"]
    assert native_epsg_error is None, native_epsg_error

    result = _run_staged_probe(
        """
import json
import os
import sys
from pathlib import Path

import laspy
import pyproj

stage = Path(os.environ["TREELINE_TEST_PDAL_STAGE"])
site = stage / "Lib" / "site-packages"
runtime = stage / "Library" / "bin"
sys.path.insert(0, str(site))
os.environ["PATH"] = os.pathsep.join([str(runtime), os.environ.get("PATH", "")])
os.environ["PROJ_NETWORK"] = "OFF"
pyproj.network.set_network_enabled(False)
pyproj_module = Path(pyproj.__file__).resolve()
pyproj_version = str(pyproj.proj_version_str)
pyproj_data = Path(pyproj.datadir.get_data_dir()).resolve()
assert (pyproj_data / "proj.db").is_file(), pyproj_data
native_proj_data = Path(os.environ["TREELINE_TEST_PDAL_PROJ_DATA"])
assert (native_proj_data / "proj.db").is_file(), native_proj_data
os.environ["PROJ_DATA"] = str(native_proj_data)
transformer = pyproj.Transformer.from_crs(
    "EPSG:4326", "EPSG:3857", always_xy=True, allow_ballpark=False, only_best=True
)
expected_x, expected_y = transformer.transform(1.0, 1.0, errcheck=True)
dll_handle = os.add_dll_directory(str(runtime))
import pdal

module_path = Path(pdal.__file__).resolve()
assert module_path.is_relative_to(site / "pdal"), module_path
pdal_info = getattr(pdal, "info", None)
assert pdal_info is not None
assert str(getattr(pdal_info, "version", "")) == "2.10.2"
reprojection_path = sys.argv[2] + ".reprojected.las"
reprojection_pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": sys.argv[1]},
        {
            "type": "filters.reprojection",
            "in_srs": "+proj=longlat +datum=WGS84 +no_defs",
            "out_srs": "+proj=merc +R=6378137 +lon_0=0 +k=1 +x_0=0 +y_0=0 +units=m +no_defs",
        },
        {"type": "writers.las", "filename": reprojection_path, "forward": "all"},
    ]
}
reprojection_processed = int(
    pdal.Pipeline(json.dumps(reprojection_pipeline)).execute()
)
assert reprojection_processed > 0
reprojected = laspy.read(reprojection_path)
reprojection_x = float(reprojected.x[21])
reprojection_y = float(reprojected.y[21])
assert abs(reprojection_x - expected_x) < 0.05, (reprojection_x, expected_x)
assert abs(reprojection_y - expected_y) < 0.05, (reprojection_y, expected_y)
pipeline = {
    "pipeline": [
        {"type": "readers.las", "filename": sys.argv[1]},
        {
            "type": "filters.smrf",
            "cell": 8.0,
            "slope": 0.2,
            "threshold": 0.45,
            "window": 16.0,
        },
        {"type": "filters.hag_nn", "count": 8, "allow_extrapolation": True},
        {
            "type": "writers.las",
            "filename": sys.argv[2],
            "extra_dims": "all",
            "forward": "all",
        },
    ]
}
processed = int(pdal.Pipeline(json.dumps(pipeline)).execute())
with laspy.open(sys.argv[2]) as reader:
    dimensions = set(reader.header.point_format.dimension_names)
    point_count = reader.header.point_count
assert processed == point_count > 0
assert "HeightAboveGround" in dimensions
assert pyproj.network.is_network_enabled() is False
print(json.dumps({
    "pdal_module": str(module_path),
    "pdal_version": str(getattr(pdal_info, "version", "unknown")),
    "pyproj_module": str(pyproj_module),
    "pyproj_proj_version": pyproj_version,
    "proj_data": str(pyproj_data),
    "pdal_reprojection": {"x": reprojection_x, "y": reprojection_y},
}, sort_keys=True), flush=True)
dll_handle.close()
""",
        input_path,
        output_path,
    )
    assert result.returncode == 0, result.stderr
    assert '"pdal_version": "2.10.2"' in result.stdout
    assert '"pyproj_proj_version":' in result.stdout
    result_payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert result_payload["pdal_reprojection"]["x"] == pytest.approx(
        111319.4908, abs=0.05
    )
    assert result_payload["pdal_reprojection"]["y"] == pytest.approx(
        111325.1429, abs=0.05
    )
