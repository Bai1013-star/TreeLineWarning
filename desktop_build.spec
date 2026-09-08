# ruff: noqa: F821
import ctypes
import os
import sys
from collections import deque
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

_SPEC_DLL_HANDLES = []


sys.path.insert(0, SPECPATH)
from desktop.build_contract import (  # noqa: E402
    validate_lazrs_runtime,
    validate_native_proj_data,
    validate_pdal_binding_abi,
    validate_pillow_runtime,
    validate_pyproj_runtime,
)


def _bootstrap_staged_sqlite() -> Path:
    """Preload the approved SQLite ABI before PyInstaller imports extensions."""

    configured = os.environ.get("TREELINE_PDAL_RUNTIME_ENV", "").strip()
    if not configured:
        raise SystemExit("TREELINE_PDAL_RUNTIME_ENV is required before native collectors run")
    sqlite_path = Path(configured).resolve() / "Library" / "bin" / "sqlite3.dll"
    if not sqlite_path.is_file():
        raise SystemExit(f"approved staged sqlite3.dll is unavailable: {sqlite_path}")
    if hasattr(os, "add_dll_directory"):
        _SPEC_DLL_HANDLES.append(os.add_dll_directory(str(sqlite_path.parent)))
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise SystemExit("the Windows build interpreter cannot preload sqlite3.dll")
    try:
        _SPEC_DLL_HANDLES.append(loader(str(sqlite_path)))
    except OSError as exc:
        raise SystemExit(f"approved staged sqlite3.dll failed to load: {sqlite_path}") from exc
    return sqlite_path

_bootstrap_staged_sqlite()

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "TreeLineWarning desktop release builds require CPython 3.11 "
        "to match libpdalpython.cp311-win_amd64.pyd"
    )

validate_pillow_runtime(python_version=sys.version_info[:2])
validate_pyproj_runtime()
validate_lazrs_runtime()

hiddenimports = (
    collect_submodules("algorithms")
    + collect_submodules("core")
    + collect_submodules("pyproj")
    + collect_submodules("cupy")
    + collect_submodules("cuda")
    + [
        "scripts",
        "scripts.serve_p3a_giro3d",
        "scripts.tree_viewer_api",
        "desktop.project_management",
        "desktop.workflow",
        "desktop.server",
        "desktop.pdal_runtime",
        "desktop.work_order",
        "plotly",
        "plotly.offline",
        "lazrs",
    ]
)


def _gpu_runtime() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Collect the pinned CuPy CUDA-component-wheel runtime for optional GPU use."""

    import importlib.metadata

    expected = {
        "cupy-cuda13x": "14.2.0",
        "cuda-pathfinder": "1.8.1",
        "nvidia-cuda-runtime": "13.3.29",
        "nvidia-cuda-nvrtc": "13.3.33",
        "nvidia-nvjitlink": "13.3.33",
    }
    observed = {}
    for name, version in expected.items():
        try:
            observed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise SystemExit(
                f"release build requires {name}=={version}; install "
                "desktop-release-gpu-requirements.txt"
            ) from exc
    mismatched = {
        name: {"expected": expected[name], "observed": version}
        for name, version in observed.items()
        if version != expected[name]
    }
    if mismatched:
        raise SystemExit(f"release GPU runtime version mismatch: {mismatched}")
    binaries = collect_dynamic_libs("nvidia")
    required = {"cudart64_13.dll", "nvrtc64_130_0.dll", "nvjitlink_130_0.dll"}
    present = {Path(source).name.casefold() for source, _destination in binaries}
    missing = sorted(name for name in required if name.casefold() not in present)
    if missing:
        raise SystemExit(f"release GPU runtime is missing required DLLs: {missing}")
    datas = (
        collect_data_files("cupy")
        + collect_data_files("cuda")
        + collect_data_files("nvidia")
    )
    return binaries, datas


gpu_binaries, gpu_datas = _gpu_runtime()

# The production PDAL stages (LAS reader/writer, SMRF, HAG-NN, and covariance
# features) are built into pdalcpp.dll.  COPC is decoded by the vendored viewer
# path, so no optional PDAL plugin is needed.  Keep this allowlist explicit so
# an environment containing incomplete optional plugin closures cannot leak
# those DLLs into a release package.
_PDAL_OPTIONAL_PLUGIN_ALLOWLIST: frozenset[str] = frozenset()


def _pdal_binaries() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Collect only the local PDAL binding and its native import closure.

    The build environment is supplied explicitly by ``TREELINE_PDAL_RUNTIME_ENV``.
    No developer path is written into the package; the runtime hook resolves all
    bundled files relative to PyInstaller's resource directory.
    """

    root_value = os.environ.get("TREELINE_PDAL_RUNTIME_ENV", "").strip()
    if not root_value:
        raise SystemExit(
            "TREELINE_PDAL_RUNTIME_ENV is required: a release must bundle the "
            "PDAL Ground/HAG runtime"
        )
    root = Path(root_value).resolve()
    bin_root = root / "Library" / "bin"
    pdal_site = root / "Lib" / "site-packages" / "pdal"
    plugin_root = root / "Lib" / "site-packages" / "bin"
    if not bin_root.is_dir() or not pdal_site.is_dir():
        raise SystemExit("TREELINE_PDAL_RUNTIME_ENV does not contain a conda PDAL runtime")
    # Let PyInstaller resolve native dependencies of all included conda
    # extensions (for example Pillow and SciPy), not only the explicit PDAL
    # import closure below.  The collected package remains self-contained.
    os.environ["PATH"] = os.pathsep.join(
        [str(bin_root), os.environ.get("PATH", "")]
    ).rstrip(os.pathsep)
    # PyInstaller itself runs from the lean desktop build environment.  Make
    # the explicitly selected portable runtime visible to its module/data
    # collectors without relying on an ambient developer PYTHONPATH.
    pdal_site_parent = str(pdal_site.parent)
    if pdal_site_parent not in sys.path:
        sys.path.append(pdal_site_parent)

    pdal_extensions = tuple(pdal_site.glob("*.pyd"))
    validate_pdal_binding_abi(
        pdal_extensions, python_version=(sys.version_info.major, sys.version_info.minor)
    )
    candidates = {path.name.lower(): path for path in bin_root.glob("*.dll")}
    candidates.update({path.name.lower(): path for path in pdal_extensions})
    queue = deque(["pdalcpp.dll", *[path.name for path in pdal_extensions]])
    seen: set[str] = set()
    native: list[tuple[str, str]] = []
    try:
        import pefile
    except ImportError as exc:  # pragma: no cover - PyInstaller provides pefile
        raise SystemExit("PyInstaller build requires pefile to collect PDAL DLLs") from exc

    while queue:
        name = queue.popleft().lower()
        if name in seen or name not in candidates:
            continue
        seen.add(name)
        source = candidates[name]
        destination = "pdal" if source.suffix.lower() == ".pyd" else "pdal_runtime"
        native.append((str(source), destination))
        binary = None
        try:
            binary = pefile.PE(str(source), fast_load=True)
            binary.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
            imports = [
                entry.dll.decode(errors="replace")
                for entry in binary.DIRECTORY_ENTRY_IMPORT
            ]
        except Exception:
            imports = []
        finally:
            if binary is not None:
                binary.close()
        queue.extend(import_name for import_name in imports if import_name.lower() in candidates)

    plugin_candidates = (*bin_root.glob("libpdal_plugin_*.dll"), *plugin_root.glob("*.dll"))
    plugin_binaries = [
        (str(path), "pdal_plugins")
        for path in plugin_candidates
        if path.name.lower() in _PDAL_OPTIONAL_PLUGIN_ALLOWLIST
    ]
    return native + plugin_binaries, collect_data_files("pdal", include_py_files=True), [
        *collect_submodules("pdal"),
        "pdal.libpdalpython",
    ]


def _required_compat_binaries() -> list[tuple[str, str]]:
    """Collect release-required codec DLLs from an explicit validated root."""

    roots = []
    configured = os.environ.get("TREELINE_COMPAT_DLL_ROOT", "").strip()
    if configured:
        roots.append(Path(configured).resolve())
    pdal_root = os.environ.get("TREELINE_PDAL_RUNTIME_ENV", "").strip()
    if pdal_root:
        roots.append(Path(pdal_root).resolve() / "Library" / "bin")
    required = ("openjp2.dll", "zlib-ng2.dll")
    collected = []
    for name in required:
        source = next(
            (root / name for root in roots if (root / name).is_file()),
            None,
        )
        if source is None:
            raise SystemExit(
                f"release build requires {name}; set TREELINE_COMPAT_DLL_ROOT "
                "to a validated portable runtime"
            )
        collected.append((str(source), "."))
    return collected


def _pdal_proj_data() -> list[tuple[str, str]]:
    """Collect the PROJ database paired with the staged native PDAL DLL."""

    configured = os.environ.get("TREELINE_PDAL_PROJ_DATA_ROOT", "").strip()
    if not configured:
        raise SystemExit(
            "TREELINE_PDAL_PROJ_DATA_ROOT is required: a release must bundle "
            "the PROJ database compatible with native PDAL"
        )
    data_root = validate_native_proj_data(
        Path(configured), native_runtime_root=Path(os.environ["TREELINE_PDAL_RUNTIME_ENV"])
    )
    return [(str(data_root), "pdal_proj_data")]


pdal_binaries, pdal_datas, pdal_hiddenimports = _pdal_binaries()
compat_binaries = _required_compat_binaries()
pdal_proj_datas = _pdal_proj_data()
pillow_binaries = collect_dynamic_libs("PIL")
pyproj_binaries = collect_dynamic_libs("pyproj")
pyproj_datas = collect_data_files("pyproj")
hiddenimports += pdal_hiddenimports
optional_excludes = [
    # Optional scientific/visualization backends that are not imported by the
    # desktop workflow.  Keeping them out avoids bundling GPU and Qt stacks
    # merely because the PDAL-capable build environment happens to contain them.
    "open3d",
    "open3d.cpu",
    "open3d.cuda",
    "torch",
    "torchvision",
    "tensorflow",
    "pandas",
    "pyarrow",
    "matplotlib",
    "dash",
    "PySide6",
    "shiboken6",
    "PyQt5",
    "PyQt6",
    "PySide2",
    "wx",
    "gi",
    "geopandas",
    "osgeo",
    "shapely",
    "streamlit",
    "sqlalchemy",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
]
datas = [
    ("viewer_poc/giro3d", "viewer_poc/giro3d"),
    ("scripts/serve_p3a_giro3d.py", "scripts"),
    *pdal_datas,
    *pdal_proj_datas,
    *pyproj_datas,
    *gpu_datas,
]

a = Analysis(
    ["desktop/app.py"],
    pathex=["."],
    binaries=[
        *pdal_binaries,
        *pillow_binaries,
        *pyproj_binaries,
        *compat_binaries,
        *gpu_binaries,
    ],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["desktop/pdal_runtime_hook.py"],
    excludes=optional_excludes,
    noarchive=False,
)


def _retain_staged_sqlite(entries):
    """Make every collected sqlite3.dll use the staged PDAL ABI."""

    configured = os.environ.get("TREELINE_PDAL_RUNTIME_ENV", "").strip()
    approved = (
        Path(configured).resolve() / "Library" / "bin" / "sqlite3.dll"
        if configured
        else None
    )
    retained = []
    found = False
    for entry in entries:
        destination, source, *metadata = entry
        if Path(destination).name.casefold() != "sqlite3.dll":
            retained.append(entry)
            continue
        if approved is None:
            raise SystemExit("TREELINE_PDAL_RUNTIME_ENV is required for sqlite3.dll collection")
        found = True
        retained.append((destination, str(approved), *metadata))
    if not found:
        raise SystemExit("release TOC does not contain the approved staged sqlite3.dll")
    return retained


a.binaries = _retain_staged_sqlite(a.binaries)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TreeLineWarning", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TreeLineWarning")
