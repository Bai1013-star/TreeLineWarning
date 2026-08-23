# -*- mode: python ; coding: utf-8 -*-
import os
import sys
from collections import deque
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

sys.path.insert(0, SPECPATH)
from desktop.build_contract import validate_pdal_binding_abi, validate_pillow_runtime

if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        "TreeLineWarning desktop release builds require CPython 3.11 "
        "to match libpdalpython.cp311-win_amd64.pyd"
    )

validate_pillow_runtime(python_version=sys.version_info[:2])

hiddenimports = (
    collect_submodules("algorithms")
    + collect_submodules("core")
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
    ]
)


def _pdal_binaries() -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
    """Collect only the local PDAL binding and its native import closure.

    The build environment is supplied explicitly by ``TREELINE_PDAL_RUNTIME_ENV``.
    No developer path is written into the package; the runtime hook resolves all
    bundled files relative to PyInstaller's resource directory.
    """

    root_value = os.environ.get("TREELINE_PDAL_RUNTIME_ENV", "").strip()
    if not root_value:
        raise SystemExit(
            "TREELINE_PDAL_RUNTIME_ENV is required: a release must bundle the PDAL Ground/HAG runtime"
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
        try:
            binary = pefile.PE(str(source), fast_load=True)
            binary.parse_data_directories(
                directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
            )
            imports = [entry.dll.decode(errors="replace") for entry in binary.DIRECTORY_ENTRY_IMPORT]
        except Exception:
            imports = []
        queue.extend(import_name for import_name in imports if import_name.lower() in candidates)

    plugin_binaries = [
        (str(path), "pdal_plugins")
        for path in (*bin_root.glob("libpdal_plugin_*.dll"), *plugin_root.glob("*.dll"))
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


pdal_binaries, pdal_datas, pdal_hiddenimports = _pdal_binaries()
compat_binaries = _required_compat_binaries()
pillow_binaries = collect_dynamic_libs("PIL")
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
    "cupy",
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
optional_gpu_dlls = {
    "cublas64_12.dll",
    "cublaslt64_12.dll",
    "cudart64_12.dll",
    "cusolver64_11.dll",
    "cusparse64_12.dll",
    "nvjitlink_120_0.dll",
}
datas = [
    ("viewer_poc", "viewer_poc"),
    ("scripts/serve_p3a_giro3d.py", "scripts"),
    ("models/species_growth.json", "models"),
    ("models/temporal_attention.npz", "models"),
    *pdal_datas,
]

a = Analysis(
    ["desktop/app.py"], pathex=["."], binaries=[*pdal_binaries, *pillow_binaries, *compat_binaries], datas=datas, hiddenimports=hiddenimports,
    hookspath=[], hooksconfig={}, runtime_hooks=["desktop/pdal_runtime_hook.py"],
    excludes=optional_excludes, noarchive=False,
)
# Some conda scientific wheels advertise optional CUDA imports while importing
# their CPU paths.  The desktop workflow is CPU/PDAL-only; remove only these
# known optional GPU libraries from the generated TOC, leaving the PDAL import
# closure collected above intact.
a.binaries = [
    entry for entry in a.binaries
    if Path(entry[0]).name.lower() not in optional_gpu_dlls
]
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="TreeLineWarning", console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="TreeLineWarning")
