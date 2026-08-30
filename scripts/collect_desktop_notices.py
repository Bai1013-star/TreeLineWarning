"""Collect the exact third-party notices used by the Windows desktop package.

This is an inventory tool, not a license-clearance tool.  It records the
license files that are present in the selected Python environment, hashes the
actual native and browser payloads, and keeps upstream source records separate
from claims about the provenance of historical binaries or generated bundles.

Run it from the repository with the release interpreter::

    .venv311\\Scripts\\python.exe scripts\\collect_desktop_notices.py

The generated text file contains the local license texts verbatim between
delimiters.  The JSON file is intended for release review and CI checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import UTC, datetime
from email.parser import Parser
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENV_ROOT = REPO_ROOT / ".venv311"
DEFAULT_RUNTIME_ROOT = REPO_ROOT / ".build-pdal-runtime-final-20260830"
DEFAULT_PROJ_DATA_ROOT = REPO_ROOT / ".build-pdal-proj-data-final-20260830-02"
DEFAULT_VIEWER_ROOT = REPO_ROOT / "viewer_poc" / "giro3d"
DEFAULT_WEBVIEW_ROOT = (
    REPO_ROOT
    / "acceptance_evidence"
    / "final_delivery_20260830"
    / "webview2_offline_prerequisite"
)
DEFAULT_OUTPUT_TEXT = (
    REPO_ROOT / "acceptance_evidence" / "final_delivery_20260830" / "THIRD_PARTY_NOTICES.txt"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT / "acceptance_evidence" / "final_delivery_20260830" / "THIRD_PARTY_NOTICES.json"
)
DEFAULT_CONDA_MATCH_AUDIT = (
    REPO_ROOT / "acceptance_evidence" / "final_delivery_20260830" / "sol_native_conda_matches.json"
)

_LICENSE_NAME = re.compile(r"(?:license|licence|copying|notice)", re.IGNORECASE)
_BINARY_SUFFIXES = {".dll", ".pyd", ".so", ".dylib"}
_MISSING = "MISSING_LOCAL_ARTIFACT"

# The first group is the runtime closure of the desktop application.  The
# second group is used only to build the package, but its notices remain useful
# when auditing the exact release environment.
RUNTIME_DISTRIBUTIONS = (
    "Pillow",
    "laspy",
    "lazrs",
    "numpy",
    "openpyxl",
    "pyproj",
    "pyshp",
    "scipy",
    "plotly",
    "narwhals",
    "pywebview",
    "pythonnet",
    "clr-loader",
    "proxy_tools",
    "bottle",
    "typing_extensions",
    "cffi",
    "pycparser",
    "packaging",
    "certifi",
    "et_xmlfile",
)
BUILD_DISTRIBUTIONS = (
    "PyInstaller",
    "pyinstaller-hooks-contrib",
    "pefile",
    "altgraph",
    "pywin32-ctypes",
    "setuptools",
)

# These are the files that the current desktop spec can collect as native
# Python extensions.  They are recorded as payload candidates; the final
# PyInstaller TOC remains the authority for the exact build output.
_PYTHON_BINARY_ROOTS: dict[str, tuple[str, ...]] = {
    "pillow": ("PIL",),
    "numpy": ("numpy", "numpy.libs"),
    "scipy": ("scipy", "scipy.libs"),
    "pyproj": ("pyproj", "pyproj.libs"),
    "lazrs": ("lazrs",),
    "pythonnet": ("pythonnet",),
    "clr-loader": ("clr_loader",),
    "cffi": ("_cffi_backend*.pyd",),
}

# Package metadata was read from the exact npm tarballs on 2026-08-30.  The
# local bundle has no lockfile or reproducible build record, so each mapping is
# deliberately labelled as an upstream candidate rather than a proof that the
# bytes came from that tarball.
BROWSER_COMPONENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "Giro3D",
        "package": "@giro3d/giro3d",
        "version": "2.0.3",
        "license": "MIT",
        "artifact_paths": ("vendor/giro3d.bundle.js",),
        "registry_metadata": "https://registry.npmjs.org/@giro3d%2fgiro3d/2.0.3",
        "tarball": "https://registry.npmjs.org/@giro3d/giro3d/-/giro3d-2.0.3.tgz",
        "integrity": (
            "sha512-dQXeoUKH1ZCdjOYFM7rqG9yi3zW2CfmuB5oWTkmxqDnoIAYnYkGKpd9v5EkUoWVwrmYErMQgOD3ONHQI1KThMg=="
        ),
        "shasum": "60862b4689536ebcd55ee39eac8586637d46e87d",
        "upstream_source": "https://gitlab.com/giro3d/giro3d/-/tree/v2.0.3",
        "license_source": "https://gitlab.com/giro3d/giro3d/-/blob/v2.0.3/LICENSE",
    },
    {
        "name": "Three.js",
        "package": "three",
        "version": "0.180.0",
        "license": "MIT",
        "artifact_paths": ("vendor/giro3d.bundle.js",),
        "registry_metadata": "https://registry.npmjs.org/three/0.180.0",
        "tarball": "https://registry.npmjs.org/three/-/three-0.180.0.tgz",
        "integrity": (
            "sha512-o+qycAMZrh+TsE01GqWUxUIKR1AL0S8pq7zDkYOQw8GqfX8b8VoCKYUoHbhiX5j+7hr8XsuHDVU6+gkQJQKg9w=="
        ),
        "shasum": "b930cabfb524f6d36bf63e874b4d866888b50487",
        "upstream_source": "https://github.com/mrdoob/three.js/tree/r180",
        "license_source": "https://github.com/mrdoob/three.js/blob/r180/LICENSE",
    },
    {
        "name": "OpenLayers",
        "package": "ol",
        "version": "10.10.0",
        "license": "BSD-2-Clause",
        "artifact_paths": ("vendor/giro3d.bundle.js",),
        "registry_metadata": "https://registry.npmjs.org/ol/10.10.0",
        "tarball": "https://registry.npmjs.org/ol/-/ol-10.10.0.tgz",
        "integrity": (
            "sha512-tLPKn6zl+6uWdPufYlqG/lQzuVUTVmfwahQqVr5+wZNyZecyAtIhMTyOtKpu7ooNDLY2sEjKZNXw9HL+sOjC1A=="
        ),
        "shasum": "24b843f453f6abee87e3c83ec58177091daee4dc",
        "upstream_source": "https://github.com/openlayers/openlayers/tree/v10.10.0",
        "license_source": "https://github.com/openlayers/openlayers/blob/v10.10.0/LICENSE.md",
    },
    {
        "name": "proj4js",
        "package": "proj4",
        "version": "2.21.0",
        "license": "MIT",
        "artifact_paths": ("vendor/giro3d.bundle.js",),
        "registry_metadata": "https://registry.npmjs.org/proj4/2.21.0",
        "tarball": "https://registry.npmjs.org/proj4/-/proj4-2.21.0.tgz",
        "integrity": (
            "sha512-33HfDftqw8kY+Cl1dcL16SJuqTSzYxmz4re7Nmhk+fs1/N1fFrAkkF579msQTR/4fLeJVSNne8/gpoCz0fk1uw=="
        ),
        "shasum": "dfe137658fd8c129818e800a95395efe531fa26b",
        "upstream_source": "https://github.com/proj4js/proj4js/tree/2.21.0",
        "license_source": "https://github.com/proj4js/proj4js/blob/2.21.0/LICENSE.md",
    },
    {
        "name": "copc.js",
        "package": "copc",
        "version": "0.0.8",
        "license": "MIT",
        "artifact_paths": ("vendor/giro3d.bundle.js",),
        "registry_metadata": "https://registry.npmjs.org/copc/0.0.8",
        "tarball": "https://registry.npmjs.org/copc/-/copc-0.0.8.tgz",
        "integrity": (
            "sha512-OWjfOZKhh7EkSYIMST/yxmegMwCR5Gye3Koq3ov1+oHFozBbPwlK1bsSWB9aNdJrXElaTh374bNhOSPxx1xSog=="
        ),
        "shasum": "10ed04ad5185709a0d1800cffc4d49da44772357",
        "upstream_source": "https://github.com/connormanning/copc.js",
        "license_source": "https://github.com/connormanning/copc.js/blob/master/LICENSE",
    },
    {
        "name": "laz-perf",
        "package": "laz-perf",
        "version": "0.0.7",
        "license": "Apache-2.0",
        "artifact_paths": ("vendor/laz-perf/laz-perf.wasm",),
        "registry_metadata": "https://registry.npmjs.org/laz-perf/0.0.7",
        "tarball": "https://registry.npmjs.org/laz-perf/-/laz-perf-0.0.7.tgz",
        "integrity": (
            "sha512-2xRqm/f/2UqDS5qqkjOyDb6uVIjkVw6SGmQlMoTQliVWkZhPfIbUJTubUl3mTloaWqKcjlS/urmP1CYoxelsEg=="
        ),
        "shasum": "949da3e04134e0a9fc56a0a3a909d84be484bf5a",
        "upstream_source": "https://github.com/hobuinc/laz-perf",
        "license_source": "https://raw.githubusercontent.com/hobuinc/laz-perf/master/COPYING",
    },
)

# These references describe the component families observed in the staged
# runtime.  A local conda byte match can support the source record for a row,
# while license and redistribution review remain separate.  Unmatched rows
# stay UNVERIFIED_SOURCE_PROVENANCE even when a version resource or the PDAL
# probe supplied a useful observed version.
_NATIVE_COMPONENTS: tuple[dict[str, Any], ...] = (
    {
        "name": "PDAL",
        "observed_version": "2.10.2",
        "files": ("Library/bin/pdalcpp.dll",),
        "license_candidate": "BSD-3-Clause",
        "upstream_source": "https://github.com/PDAL/PDAL/tree/2.10.2",
        "license_source": "https://github.com/PDAL/PDAL/blob/2.10.2/LICENSE.md",
    },
    {
        "name": "GDAL",
        "observed_version": "3.13.1",
        "files": ("Library/bin/gdal.dll",),
        "license_candidate": "MIT/X",
        "upstream_source": "https://github.com/OSGeo/gdal/tree/v3.13.1",
        "license_source": "https://github.com/OSGeo/gdal/blob/v3.13.1/LICENSE.TXT",
    },
    {
        "name": "PROJ",
        "observed_version": "9.8.1",
        "files": ("Library/bin/proj_9.dll",),
        "license_candidate": "MIT",
        "upstream_source": "https://github.com/OSGeo/PROJ/tree/9.8.1",
        "license_source": "https://github.com/OSGeo/PROJ/blob/9.8.1/COPYING",
    },
    {
        "name": "GEOS",
        "observed_version": None,
        "files": ("Library/bin/geos.dll", "Library/bin/geos_c.dll"),
        "license_candidate": "LGPL-2.1-or-later",
        "upstream_source": "https://github.com/libgeos/geos",
        "license_source": "https://github.com/libgeos/geos/blob/main/COPYING",
    },
    {
        "name": "libgeotiff",
        "observed_version": None,
        "files": ("Library/bin/geotiff.dll",),
        "license_candidate": "MIT-style",
        "upstream_source": "https://github.com/OSGeo/libgeotiff",
        "license_source": "https://github.com/OSGeo/libgeotiff/blob/master/LICENSE",
    },
    {
        "name": "libxml2",
        "observed_version": "2.15.3",
        "files": ("Library/bin/libxml2.dll",),
        "license_candidate": "MIT",
        "upstream_source": "https://gitlab.gnome.org/GNOME/libxml2/-/tags/v2.15.3",
        "license_source": "https://gitlab.gnome.org/GNOME/libxml2/-/blob/v2.15.3/Copyright",
    },
    {
        "name": "SQLite",
        "observed_version": "3.53.3",
        "files": ("Library/bin/sqlite3.dll",),
        "license_candidate": "Public domain",
        "upstream_source": "https://www.sqlite.org/2026/sqlite-amalgamation-3530300.zip",
        "license_source": "https://www.sqlite.org/copyright.html",
    },
    {
        "name": "zlib",
        "observed_version": "1.3.2",
        "files": ("Library/bin/zlib.dll",),
        "license_candidate": "Zlib",
        "upstream_source": "https://github.com/madler/zlib/tree/v1.3.2",
        "license_source": "https://zlib.net/zlib_license.html",
    },
    {
        "name": "zstd",
        "observed_version": "1.5.7",
        "files": ("Library/bin/zstd.dll",),
        "license_candidate": "BSD-3-Clause",
        "upstream_source": "https://github.com/facebook/zstd/tree/v1.5.7",
        "license_source": "https://github.com/facebook/zstd/blob/v1.5.7/LICENSE",
    },
    {
        "name": "OpenSSL",
        "observed_version": "3.6.3",
        "files": ("Library/bin/libcrypto-3-x64.dll", "Library/bin/libssl-3-x64.dll"),
        "license_candidate": "Apache-2.0",
        "upstream_source": "https://github.com/openssl/openssl/tree/openssl-3.6.3",
        "license_source": "https://github.com/openssl/openssl/blob/openssl-3.6.3/LICENSE.txt",
    },
    {
        "name": "curl",
        "observed_version": "8.21.0",
        "files": ("Library/bin/libcurl.dll",),
        "license_candidate": "curl license (MIT-style)",
        "upstream_source": "https://github.com/curl/curl/tree/curl-8_21_0",
        "license_source": "https://github.com/curl/curl/blob/curl-8_21_0/COPYING",
    },
    {
        "name": "Expat",
        "observed_version": "2.8.1",
        "files": ("Library/bin/libexpat.dll",),
        "license_candidate": "MIT",
        "upstream_source": "https://github.com/libexpat/libexpat/tree/R_2_8_1",
        "license_source": "https://github.com/libexpat/libexpat/blob/R_2_8_1/expat/COPYING",
    },
    {
        "name": "XZ Utils / liblzma",
        "observed_version": "5.8.3",
        "files": ("Library/bin/liblzma.dll",),
        "license_candidate": "Mixed upstream licensing; review COPYING files",
        "upstream_source": "https://github.com/tukaani-project/xz/tree/v5.8.3",
        "license_source": "https://github.com/tukaani-project/xz/blob/v5.8.3/COPYING",
    },
    {
        "name": "libssh2",
        "observed_version": "1.11.1",
        "files": ("Library/bin/libssh2.dll",),
        "license_candidate": "BSD-3-Clause",
        "upstream_source": "https://github.com/libssh2/libssh2/tree/libssh2-1.11.1",
        "license_source": "https://github.com/libssh2/libssh2/blob/libssh2-1.11.1/COPYING",
    },
    {
        "name": "ICU",
        "observed_version": "78.3",
        "files": ("Library/bin/icudt78.dll", "Library/bin/icuuc78.dll"),
        "license_candidate": "Unicode-3.0",
        "upstream_source": "https://github.com/unicode-org/icu/tree/release-78-1",
        "license_source": "https://github.com/unicode-org/icu/blob/release-78-1/icu4c/LICENSE",
    },
    {
        "name": "libjpeg-turbo",
        "observed_version": "8.3.2",
        "files": ("Library/bin/jpeg8.dll",),
        "license_candidate": "BSD-3-Clause / IJG / Zlib components; review source",
        "upstream_source": "https://github.com/libjpeg-turbo/libjpeg-turbo",
        "license_source": "https://github.com/libjpeg-turbo/libjpeg-turbo/blob/main/LICENSE.md",
    },
    {
        "name": "LibTIFF",
        "observed_version": "4.7.2",
        "files": ("Library/bin/tiff.dll",),
        "license_candidate": "libtiff license",
        "upstream_source": "https://gitlab.com/libtiff/libtiff/-/tags/v4.7.2",
        "license_source": "https://gitlab.com/libtiff/libtiff/-/blob/v4.7.2/LICENSE.md",
    },
    {
        "name": "Xerces-C++",
        "observed_version": "3.3.0",
        "files": ("Library/bin/xerces-c_3_3.dll",),
        "license_candidate": "Apache-2.0",
        "upstream_source": "https://github.com/apache/xerces-c/tree/Xerces-C_3_3_0",
        "license_source": "https://github.com/apache/xerces-c/blob/Xerces-C_3_3_0/LICENSE",
    },
    {
        "name": "zlib-ng compatibility DLL",
        "observed_version": "2.3.3",
        "files": ("compat/zlib-ng2.dll",),
        "license_candidate": "Zlib",
        "upstream_source": "https://github.com/zlib-ng/zlib-ng/tree/2.3.3",
        "license_source": "https://github.com/zlib-ng/zlib-ng/blob/2.3.3/LICENSE.md",
    },
    {
        "name": "OpenJPEG compatibility DLL",
        "observed_version": None,
        "files": ("compat/openjp2.dll",),
        "license_candidate": "BSD-2-Clause candidate; exact binary source not established",
        "upstream_source": "https://github.com/uclouvain/openjpeg",
        "license_source": "https://github.com/uclouvain/openjpeg/blob/master/LICENSE.md",
    },
    {
        "name": "Microsoft Visual C++ runtime",
        "observed_version": "14.51.36231.0",
        "files": (
            "Library/bin/msvcp140.dll",
            "Library/bin/vcruntime140.dll",
            "Library/bin/vcruntime140_1.dll",
            "Library/bin/vcomp140.dll",
        ),
        "license_candidate": "Microsoft redistributable terms; review installer/source package",
        "upstream_source": "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist",
        "license_source": "https://visualstudio.microsoft.com/license-terms/",
    },
    {
        "name": "Universal CRT API-set files",
        "observed_version": "10.0.26100.4654",
        "files": ("Library/bin/api-ms-win-crt-*.dll",),
        "license_candidate": "Microsoft Windows component terms; review redistribution source",
        "upstream_source": "https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist",
        "license_source": "https://visualstudio.microsoft.com/license-terms/",
    },
)


def _normalise_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_path(path: Path) -> str:
    """Render a path portably while retaining an absolute path outside repo."""

    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _metadata(path: Path) -> Any:
    metadata_path = path / "METADATA"
    if not metadata_path.is_file():
        return None
    try:
        return Parser().parsestr(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError):
        return None


def _site_packages(venv_root: Path) -> Path:
    candidates = (
        venv_root / "Lib" / "site-packages",
        venv_root / "lib" / "python3.11" / "site-packages",
        venv_root / "lib" / "site-packages",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _license_paths(dist_info: Path, metadata: Any) -> list[Path]:
    candidates: set[Path] = set()
    declared = metadata.get_all("License-File", []) if metadata is not None else []
    for value in declared:
        relative = Path(str(value).replace("\\", "/"))
        candidates.update((dist_info / relative, dist_info / "licenses" / relative))
    if dist_info.is_dir():
        for path in dist_info.rglob("*"):
            if path.is_file() and not path.is_symlink() and _LICENSE_NAME.search(path.name):
                candidates.add(path)
    return sorted(path for path in candidates if path.is_file() and not path.is_symlink())


def _project_urls(metadata: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if metadata is None:
        return result
    homepage = metadata.get("Home-page")
    if homepage:
        result["Homepage"] = str(homepage).strip()
    for value in metadata.get_all("Project-URL", []):
        label, separator, url = str(value).partition(",")
        if separator and url.strip():
            result[label.strip()] = url.strip()
    return dict(sorted(result.items()))


def _binary_files(site_packages: Path, normalised_name: str) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for root in _PYTHON_BINARY_ROOTS.get(normalised_name, ()):
        if any(character in root for character in "*?["):
            paths.update(site_packages.glob(root))
            continue
        candidate = site_packages / root
        if candidate.is_file() and candidate.suffix.casefold() in _BINARY_SUFFIXES:
            paths.add(candidate)
        elif candidate.is_dir():
            paths.update(
                path
                for path in candidate.rglob("*")
                if path.is_file()
                and not path.is_symlink()
                and path.suffix.casefold() in _BINARY_SUFFIXES
            )
    return [
        {
            "path": _repo_path(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(paths)
    ]


def _distribution_index(site_packages: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for dist_info in sorted(site_packages.glob("*.dist-info")):
        metadata = _metadata(dist_info)
        name = metadata.get("Name") if metadata is not None else None
        if name:
            result[_normalise_name(str(name))] = dist_info
    return result


def collect_python_distributions(venv_root: Path) -> list[dict[str, Any]]:
    """Collect selected runtime and build distributions from one site-packages."""

    site_packages = _site_packages(venv_root.resolve())
    index = _distribution_index(site_packages)
    records: list[dict[str, Any]] = []
    for classification, names in (
        ("runtime", RUNTIME_DISTRIBUTIONS),
        ("build_tool", BUILD_DISTRIBUTIONS),
    ):
        for requested_name in names:
            key = _normalise_name(requested_name)
            dist_info = index.get(key)
            if dist_info is None:
                records.append(
                    {
                        "requested_name": requested_name,
                        "classification": classification,
                        "status": _MISSING,
                        "license_files": [],
                        "binary_files": [],
                    }
                )
                continue
            metadata = _metadata(dist_info)
            license_files = []
            for path in _license_paths(dist_info, metadata):
                raw = path.read_bytes()
                license_files.append(
                    {
                        "path": _repo_path(path),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
            record_file = dist_info / "RECORD"
            record = {
                "requested_name": requested_name,
                "name": str(metadata.get("Name", requested_name)) if metadata else requested_name,
                "version": str(metadata.get("Version", "unknown")) if metadata else "unknown",
                "classification": classification,
                "dist_info_path": _repo_path(dist_info),
                "license_expression": (
                    str(metadata.get("License-Expression"))
                    if metadata and metadata.get("License-Expression")
                    else None
                ),
                "metadata_license_summary": (
                    str(metadata.get("License")).strip().splitlines()[0][:240]
                    if metadata and metadata.get("License")
                    else None
                ),
                "project_urls": _project_urls(metadata),
                "record_file": (
                    {
                        "path": _repo_path(record_file),
                        "bytes": record_file.stat().st_size,
                        "sha256": _sha256(record_file),
                    }
                    if record_file.is_file()
                    else None
                ),
                "license_files": license_files,
                "binary_files": _binary_files(site_packages, key),
                "status": (
                    "LOCAL_LICENSE_TEXT_FOUND"
                    if license_files
                    else "LICENSE_TEXT_NOT_FOUND_LOCALLY"
                ),
            }
            direct_url = dist_info / "direct_url.json"
            if direct_url.is_file():
                try:
                    record["direct_url"] = json.loads(direct_url.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    record["direct_url"] = {"status": "UNREADABLE"}
            records.append(record)
    return records


def _file_record(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {"path": _repo_path(path)}
    if not path.is_file() or path.is_symlink():
        record.update({"status": _MISSING, "bytes": None, "sha256": None})
        return record
    actual = _sha256(path)
    record.update(
        {
            "status": "PRESENT",
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    )
    if expected_sha256 is not None:
        record["manifest_sha256"] = expected_sha256
        record["hash_matches_manifest"] = actual.casefold() == expected_sha256.casefold()
    return record


def _sqlite_layout(path: Path) -> int | None:
    """Read a PROJ layout value in a child process to avoid DLL collisions."""

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sqlite3
import sys

connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
try:
    value = connection.execute(
        "SELECT value FROM metadata "
        "WHERE key = 'DATABASE.LAYOUT.VERSION.MINOR'"
    ).fetchone()
finally:
    connection.close()
print(json.dumps(int(value[0]) if value else None))
""",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if probe.returncode != 0:
        return None
    try:
        return json.loads(probe.stdout.strip().splitlines()[-1])
    except (IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _browser_assets(viewer_root: Path) -> list[dict[str, Any]]:
    if not viewer_root.is_dir():
        return []
    return [
        _file_record(path)
        for path in sorted(viewer_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]


def _browser_records(viewer_root: Path) -> list[dict[str, Any]]:
    assets = {record["path"]: record for record in _browser_assets(viewer_root)}
    records: list[dict[str, Any]] = []
    for component in BROWSER_COMPONENTS:
        item = dict(component)
        item["artifact_paths"] = [
            _repo_path(viewer_root / relative) for relative in component["artifact_paths"]
        ]
        item["artifact_files"] = [
            assets.get(path, {"path": path, "status": _MISSING}) for path in item["artifact_paths"]
        ]
        item["status"] = "UNVERIFIED_SOURCE_PROVENANCE"
        item["status_reason"] = (
            "The vendored bundle/WASM has no adjacent lockfile or reproducible build record; "
            "registry metadata is an upstream candidate only."
        )
        for artifact in item["artifact_files"]:
            artifact_path = REPO_ROOT / artifact["path"]
            if (
                component["package"] == "three"
                and artifact_path.name == "giro3d.bundle.js"
                and artifact.get("status") == "PRESENT"
            ):
                raw = artifact_path.read_bytes()
                marker = b"Bundled license information:"
                marker_index = raw.rfind(marker)
                if marker_index >= 0:
                    item["inline_notice"] = raw[marker_index:].decode(
                        "utf-8", errors="replace"
                    ).splitlines()[:4]
        records.append(item)
    return records


def _load_manifest(runtime_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = runtime_root / "native_runtime_manifest.json"
    if not manifest_path.is_file():
        return (
            {
                "path": _repo_path(manifest_path),
                "status": _MISSING,
                "sha256": None,
            },
            {"files": [], "compatibility_files": [], "versions": {}},
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_record = _file_record(manifest_path)
    rows = [*manifest.get("files", []), *manifest.get("compatibility_files", [])]
    payload = []
    for row in sorted(rows, key=lambda item: str(item.get("destination_relative", ""))):
        relative = str(row.get("destination_relative", ""))
        payload.append(
            {
                **row,
                "path": _repo_path(runtime_root / Path(relative)),
                "file": _file_record(
                    runtime_root / Path(relative), expected_sha256=row.get("sha256")
                ),
            }
        )
    manifest_record["status"] = "PRESENT"
    return (
        manifest_record,
        {
            "versions": manifest.get("versions", {}),
            "files": payload,
            "compatibility_files": [item for item in payload if item.get("license_status")],
            "optional_plugin_policy": manifest.get("payload", {}).get("optional_plugin_policy"),
        },
    )


def _proj_database_inventory(
    runtime_root: Path,
    venv_root: Path,
    proj_data_root: Path | None = None,
) -> dict[str, Any]:
    """Record native and wheel PROJ databases without loading native libraries."""

    staged = sorted(runtime_root.rglob("proj.db")) if runtime_root.is_dir() else []
    snapshot: dict[str, Any] = {"status": _MISSING}
    if proj_data_root is not None:
        proj_data_root = proj_data_root.resolve()
        snapshot_manifest = proj_data_root / "proj_data_manifest.json"
        snapshot = {
            "root": _repo_path(proj_data_root),
            "manifest": _file_record(snapshot_manifest),
            "status": "PRESENT" if snapshot_manifest.is_file() else _MISSING,
        }
        if snapshot_manifest.is_file():
            try:
                snapshot["manifest_payload"] = json.loads(
                    snapshot_manifest.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                snapshot["status"] = "UNREADABLE"
        snapshot_data_root = proj_data_root / "Library" / "share" / "proj"
        if snapshot_data_root.is_dir():
            staged = sorted(snapshot_data_root.rglob("proj.db"))
    site_packages = _site_packages(venv_root.resolve())
    wheel_candidates = sorted(site_packages.glob("pyproj*/proj_dir/share/proj/proj.db"))
    result = {
        "snapshot": snapshot,
        "staged_proj_db": _file_record(staged[0]) if staged else None,
        "pyproj_wheel_proj_db": _file_record(wheel_candidates[0])
        if wheel_candidates
        else None,
        "status": (
            "NATIVE_PROJ_DATABASE_PRESENT"
            if staged
            else "NATIVE_PROJ_DATABASE_NOT_STAGED"
        ),
    }
    for key in ("staged_proj_db", "pyproj_wheel_proj_db"):
        item = result[key]
        if item is None or item.get("status") != "PRESENT":
            continue
        path = Path(item["path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        item["layout_minor"] = _sqlite_layout(path)
    return result


def _conda_match_inventory(
    runtime_root: Path, audit_path: Path = DEFAULT_CONDA_MATCH_AUDIT
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Verify local conda byte matches recorded by the bounded native audit."""

    audit_path = audit_path.resolve()
    if not audit_path.is_file():
        return (
            {"path": _repo_path(audit_path), "status": _MISSING, "matches": []},
            {},
        )
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return (
            {"path": _repo_path(audit_path), "status": "UNREADABLE", "matches": []},
            {},
        )
    matches_by_target: dict[str, list[dict[str, Any]]] = {}
    package_records: list[dict[str, Any]] = []
    for package in audit.get("packages", []):
        package_root = Path(str(package.get("package_root", ""))).resolve()
        metadata = package.get("metadata", {})
        repodata = metadata.get("repodata_record.json", {})
        archive = package_root.parent / str(repodata.get("fn", ""))
        archive_record = _file_record(archive) if archive.name else {
            "path": str(archive),
            "status": _MISSING,
        }
        archive_hash_matches = bool(
            archive_record.get("status") == "PRESENT"
            and str(archive_record.get("sha256", "")).casefold()
            == str(repodata.get("sha256", "")).casefold()
        )
        licenses = [
            _file_record(Path(str(path)))
            for path in package.get("license_paths", [])
        ]
        package_record = {
            "package_root": str(package_root),
            "name": repodata.get("name") or metadata.get("index.json", {}).get("name"),
            "version": repodata.get("version") or metadata.get("index.json", {}).get("version"),
            "build": repodata.get("build") or metadata.get("index.json", {}).get("build"),
            "license": repodata.get("license") or metadata.get("index.json", {}).get("license"),
            "repodata": {
                key: repodata.get(key)
                for key in ("channel", "url", "sha256", "fn", "subdir")
                if repodata.get(key) is not None
            },
            "archive": archive_record,
            "archive_hash_matches_repodata": archive_hash_matches,
            "license_files": licenses,
            "matches": [],
        }
        for match in package.get("matched", []):
            target_relative = str(match.get("target", "")).replace("\\", "/")
            source = Path(str(match.get("source", ""))).resolve()
            target = runtime_root / Path(target_relative)
            observed_hash = str(match.get("sha256", "")).casefold()
            source_hash = _sha256(source) if source.is_file() else None
            target_hash = _sha256(target) if target.is_file() else None
            verified = bool(
                archive_hash_matches
                and source_hash
                and target_hash
                and source_hash.casefold() == observed_hash
                and target_hash.casefold() == observed_hash
            )
            item = {
                "target": target_relative,
                "source": str(source),
                "sha256": match.get("sha256"),
                "source_hash_matches_audit": source_hash is not None
                and source_hash.casefold() == observed_hash,
                "target_hash_matches_audit": target_hash is not None
                and target_hash.casefold() == observed_hash,
                "status": (
                    "LOCAL_CACHED_CONDA_PACKAGE_BYTE_MATCH"
                    if verified
                    else "UNVERIFIED_SOURCE_PROVENANCE"
                ),
            }
            package_record["matches"].append(item)
            matches_by_target.setdefault(target_relative, []).append(
                {**item, "package": package_record}
            )
        package_records.append(package_record)
    return (
        {
            "path": _repo_path(audit_path),
            "status": "PRESENT",
            "record": _file_record(audit_path),
            "target_count": audit.get("target_count"),
            "matched_count": audit.get("matched_count"),
            "unmatched": audit.get("unmatched", []),
            "packages": package_records,
        },
        matches_by_target,
    )


def _native_records(
    native_payload: dict[str, Any],
    runtime_root: Path,
    conda_matches: dict[str, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    conda_matches = conda_matches or {}
    file_map = {
        item.get("destination_relative"): item for item in native_payload.get("files", [])
    }
    records: list[dict[str, Any]] = []
    for component in _NATIVE_COMPONENTS:
        item = dict(component)
        expanded_paths: list[str] = []
        for pattern in component["files"]:
            if "*" in pattern:
                expanded_paths.extend(
                    path for path in file_map if Path(str(path)).match(pattern)
                )
            else:
                expanded_paths.append(pattern)
        payload_files = [
            file_map.get(
                relative,
                {
                    "path": _repo_path(runtime_root / relative),
                    "status": _MISSING,
                },
            )
            for relative in sorted(expanded_paths)
        ]
        for payload_file in payload_files:
            relative = payload_file.get("destination_relative")
            candidates = conda_matches.get(str(relative), [])
            if candidates:
                payload_file["source_provenance"] = candidates
        item["payload_files"] = payload_files
        matched_payloads = all(
            payload_file.get("source_provenance")
            and any(
                candidate.get("status") == "LOCAL_CACHED_CONDA_PACKAGE_BYTE_MATCH"
                for candidate in payload_file["source_provenance"]
            )
            for payload_file in payload_files
        )
        item["status"] = (
            "LOCAL_CACHED_CONDA_PACKAGE_BYTE_MATCH"
            if payload_files and matched_payloads
            else "UNVERIFIED_SOURCE_PROVENANCE"
        )
        item["status_reason"] = (
            "Every selected payload byte matches a local cached conda package record; "
            "license and redistribution terms still require release/legal review."
            if item["status"] == "LOCAL_CACHED_CONDA_PACKAGE_BYTE_MATCH"
            else "The staged payload has no verified local package byte match; the candidate "
            "upstream URL and local hash require release/legal review."
        )
        records.append(item)
    return records


def _webview_record(webview_root: Path) -> dict[str, Any]:
    installer = webview_root / "offline_verified" / "MicrosoftEdgeWebView2RuntimeInstallerX64.exe"
    auth = webview_root / "authenticode.json"
    eula = webview_root / "official_webview2_eula.json"
    result: dict[str, Any] = {
        "installer": _file_record(installer),
        "authenticode_record": _file_record(auth),
        "eula_record": _file_record(eula),
        "status": "PREREQUISITE_NOT_EXECUTED",
        "clean_pc_verified": False,
    }
    if auth.is_file():
        try:
            result["authenticode"] = json.loads(auth.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            result["authenticode"] = {"status": "UNREADABLE"}
    if eula.is_file():
        try:
            eula_payload = json.loads(eula.read_text(encoding="utf-8"))
            html = str(eula_payload.get("evergreenHtml", ""))
            result["eula_html_sha256"] = hashlib.sha256(html.encode("utf-8")).hexdigest()
            result["eula_html_present"] = bool(html)
        except (OSError, json.JSONDecodeError):
            result["eula_html_present"] = False
    result["source_records"] = [
        "https://go.microsoft.com/fwlink/?linkid=2124701",
        "https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution",
    ]
    return result


def collect_inventory(
    *,
    venv_root: Path = DEFAULT_VENV_ROOT,
    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    proj_data_root: Path = DEFAULT_PROJ_DATA_ROOT,
    viewer_root: Path = DEFAULT_VIEWER_ROOT,
    webview_root: Path = DEFAULT_WEBVIEW_ROOT,
) -> dict[str, Any]:
    """Return a JSON-serialisable notice inventory for one release input set."""

    distributions = collect_python_distributions(venv_root)
    manifest_record, native_payload = _load_manifest(runtime_root.resolve())
    conda_audit, conda_matches = _conda_match_inventory(runtime_root.resolve())
    proj_data = _proj_database_inventory(
        runtime_root.resolve(), venv_root.resolve(), proj_data_root
    )
    browser = _browser_records(viewer_root.resolve())
    browser_assets = _browser_assets(viewer_root.resolve())
    webview = _webview_record(webview_root.resolve())
    native_components = _native_records(
        native_payload, runtime_root.resolve(), conda_matches
    )
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    unresolved: list[dict[str, Any]] = []
    if manifest_record["status"] == _MISSING:
        unresolved.append(
            {
                "artifact": manifest_record["path"],
                "status": _MISSING,
                "reason": "The selected materialized PDAL runtime manifest is required.",
            }
        )
    if conda_audit.get("status") != "PRESENT":
        unresolved.append(
            {
                "artifact": conda_audit.get("path"),
                "status": conda_audit.get("status"),
                "reason": (
                    "The bounded local conda byte-match audit is required to verify native "
                    "source records."
                ),
            }
        )
    elif conda_audit.get("unmatched"):
        unresolved.append(
            {
                "artifact": "native conda byte-match audit unmatched files",
                "status": "UNVERIFIED_SOURCE_PROVENANCE",
                "reason": "These staged files were not matched to a local cached conda package: "
                + ", ".join(map(str, conda_audit["unmatched"])),
            }
        )
    if proj_data["status"] != "NATIVE_PROJ_DATABASE_PRESENT":
        unresolved.append(
            {
                "artifact": "native PDAL PROJ database",
                "status": proj_data["status"],
                "reason": (
                    "The staged native PROJ data snapshot has no compatible proj.db; the "
                    "available pyproj wheel database must not be assumed compatible with "
                    "the native PROJ ABI."
                ),
            }
        )
    if not browser_assets:
        unresolved.append(
            {
                "artifact": _repo_path(viewer_root),
                "status": _MISSING,
                "reason": (
                    "The release spec collects viewer_poc/giro3d; no viewer files were found."
                ),
            }
        )
    if webview["installer"]["status"] == _MISSING:
        unresolved.append(
            {
                "artifact": webview["installer"]["path"],
                "status": _MISSING,
                "reason": "The signed standalone WebView2 prerequisite record is required.",
            }
        )
    missing_distributions = [
        item["requested_name"] for item in distributions if item.get("status") == _MISSING
    ]
    unresolved.extend(
        {
            "artifact": name,
            "status": _MISSING,
            "reason": "Selected release environment does not contain this distribution metadata.",
        }
        for name in missing_distributions
    )
    unresolved.extend(
        {
            "artifact": item["name"],
            "status": "LICENSE_TEXT_NOT_FOUND_LOCALLY",
            "reason": (
                "Distribution metadata declares a license, but no exact license or notice text "
                "was present under its dist-info directory."
            ),
        }
        for item in distributions
        if item.get("status") == "LICENSE_TEXT_NOT_FOUND_LOCALLY"
    )
    if not python_license.is_file():
        unresolved.append(
            {
                "artifact": _repo_path(python_license),
                "status": _MISSING,
                "reason": "The CPython runtime license text was not found next to base_prefix.",
            }
        )
    inventory = {
        "schema": 1,
        "inventory_kind": "desktop-third-party-notices",
        "generated_utc": datetime.now(UTC).isoformat(),
        "disclaimer": (
            "Engineering inventory only; not legal clearance. Verify every included artifact, "
            "source, license, and redistribution condition before shipping."
        ),
        "release_inputs": {
            "venv_root": _repo_path(venv_root),
            "runtime_root": _repo_path(runtime_root),
            "proj_data_root": _repo_path(proj_data_root),
            "viewer_root": _repo_path(viewer_root),
            "webview_root": _repo_path(webview_root),
            "build_spec": _repo_path(REPO_ROOT / "desktop_build.spec"),
        },
        "python": {
            "interpreter": _repo_path(Path(sys.executable)),
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "license": _file_record(python_license),
            "distributions": distributions,
        },
        "browser": {
            "assets": browser_assets,
            "components": browser,
            "license_corrections": [
                {
                    "component": "laz-perf@0.0.7",
                    "superseded_documentation_claim": "LGPL-2.1",
                    "registry_and_source_license": "Apache-2.0",
                    "source": "https://raw.githubusercontent.com/hobuinc/laz-perf/master/js/package.json",
                },
                {
                    "component": "copc@0.0.8",
                    "superseded_documentation_claim": "ISC",
                    "registry_license": "MIT",
                    "source": "https://github.com/connormanning/copc.js/blob/master/package.json",
                },
            ],
        },
        "native": {
            "manifest": manifest_record,
            "conda_matches": conda_audit,
            "versions": native_payload.get("versions", {}),
            "proj_data": proj_data,
            "optional_plugin_policy": native_payload.get("optional_plugin_policy"),
            "files": native_payload.get("files", []),
            "components": native_components,
        },
        "webview2_prerequisite": webview,
        "unresolved_provenance": unresolved
        + [
            {
                "artifact": component["name"],
                "status": component["status"],
                "reason": component["status_reason"],
            }
            for component in native_components
        ]
        + [
            {
                "artifact": component["name"],
                "status": component["status"],
                "reason": component["status_reason"],
            }
            for component in browser
        ],
    }
    _, native_license_unresolved = _native_license_text_sections(inventory)
    inventory["unresolved_provenance"].extend(native_license_unresolved)
    return inventory


def _inventory_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _conda_package_identity(package: dict[str, Any]) -> str:
    package_root = Path(str(package.get("package_root", ""))).name
    name = str(package.get("name") or package_root or "unknown")
    version = str(package.get("version") or "unknown")
    build = str(package.get("build") or "unknown")
    return f"conda:{name} {version} {build}"


def _native_license_text_sections(
    inventory: dict[str, Any],
) -> tuple[list[tuple[str, str, bytes]], list[dict[str, Any]]]:
    """Return validated cached-conda license bytes and explicit local gaps."""

    conda_audit = inventory.get("native", {}).get("conda_matches", {})
    packages = conda_audit.get("packages")
    if not isinstance(packages, list):
        return [], []
    sections: list[tuple[str, str, bytes]] = []
    unresolved: list[dict[str, Any]] = []
    for package in packages:
        if not isinstance(package, dict):
            continue
        identity = _conda_package_identity(package)
        license_files = package.get("license_files")
        if not isinstance(license_files, list) or not license_files:
            package["license_text_status"] = "LICENSE_TEXT_NOT_FOUND_LOCALLY"
            declared_license = str(package.get("license") or "not declared")
            if declared_license.casefold() == "blessing":
                reason = (
                    "The matched conda package has no local license/notice text in its "
                    "cached package metadata; its metadata declares `blessing` (the "
                    "SQLite public-domain declaration), which is not legal clearance."
                )
            else:
                reason = (
                    "The matched conda package has no local license/notice text in its "
                    f"cached package metadata (declared license={declared_license}); "
                    "this is not legal clearance."
                )
            unresolved.append(
                {
                    "artifact": f"{identity} license text",
                    "status": "LICENSE_TEXT_NOT_FOUND_LOCALLY",
                    "reason": reason,
                }
            )
            continue
        valid_count = 0
        package_issue = False
        for record in license_files:
            if not isinstance(record, dict):
                package_issue = True
                unresolved.append(
                    {
                        "artifact": f"{identity} license record",
                        "status": "LICENSE_TEXT_NOT_FOUND_LOCALLY",
                        "reason": "The cached package license record is not an object.",
                    }
                )
                continue
            path = _inventory_path(record.get("path"))
            if record.get("status") != "PRESENT" or path is None:
                package_issue = True
                unresolved.append(
                    {
                        "artifact": f"{identity} license: {record.get('path')}",
                        "status": record.get("status") or "LICENSE_TEXT_NOT_FOUND_LOCALLY",
                        "reason": "The cached package license/notice text is missing locally.",
                    }
                )
                continue
            if not path.is_file() or path.is_symlink():
                package_issue = True
                unresolved.append(
                    {
                        "artifact": f"{identity} license: {_repo_path(path)}",
                        "status": "MISSING_LOCAL_ARTIFACT",
                        "reason": "The cached package license/notice path is not a regular file.",
                    }
                )
                continue
            raw = path.read_bytes()
            actual_hash = hashlib.sha256(raw).hexdigest()
            expected_hash = record.get("sha256")
            expected_bytes = record.get("bytes")
            if (
                type(expected_bytes) is not int
                or expected_bytes != len(raw)
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or expected_hash.casefold() != actual_hash.casefold()
            ):
                package_issue = True
                unresolved.append(
                    {
                        "artifact": f"{identity} license: {_repo_path(path)}",
                        "status": "LICENSE_TEXT_HASH_MISMATCH",
                        "reason": (
                            "The cached package license/notice bytes no longer match the "
                            "recorded byte count and SHA-256."
                        ),
                        "recorded_bytes": expected_bytes,
                        "actual_bytes": len(raw),
                        "recorded_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                    }
                )
                continue
            valid_count += 1
            sections.append((identity, str(record["path"]), raw))
        package["license_text_status"] = (
            "LICENSE_TEXT_PARTIALLY_INVALID"
            if package_issue and valid_count
            else "LICENSE_TEXT_NOT_FOUND_LOCALLY"
            if package_issue
            else "LOCAL_LICENSE_TEXT_FOUND"
        )
    return sections, unresolved


def _license_text_sections(inventory: dict[str, Any]) -> list[tuple[str, str, bytes]]:
    sections: list[tuple[list[str], str, bytes]] = []
    seen: dict[tuple[str, str], int] = {}

    def append_section(package: str, path_value: object, raw: bytes) -> None:
        path = _inventory_path(path_value)
        if path is None:
            return
        key = (str(path.resolve()).casefold(), hashlib.sha256(raw).hexdigest())
        index = seen.get(key)
        if index is None:
            seen[key] = len(sections)
            sections.append(([package], str(path_value), raw))
        elif package not in sections[index][0]:
            sections[index][0].append(package)

    for distribution in inventory["python"]["distributions"]:
        if distribution.get("status") == _MISSING:
            continue
        for item in distribution.get("license_files", []):
            path = _inventory_path(item.get("path"))
            if path is None:
                continue
            if path.is_file():
                append_section(distribution["name"], item.get("path"), path.read_bytes())
    python_license = inventory["python"]["license"]
    if python_license.get("status") == "PRESENT":
        path = _inventory_path(python_license.get("path"))
        if path is not None and path.is_file():
            append_section("CPython", python_license.get("path"), path.read_bytes())
    proj_snapshot = inventory["native"]["proj_data"].get("snapshot", {})
    snapshot_root = proj_snapshot.get("root")
    if snapshot_root and proj_snapshot.get("status") == "PRESENT":
        snapshot_path = _inventory_path(snapshot_root)
        license_path = snapshot_path / "provenance" / "COPYING" if snapshot_path else None
        if license_path is not None and license_path.is_file():
            append_section(
                "PROJ 9.8.1 conda package",
                _repo_path(license_path),
                license_path.read_bytes(),
            )
    native_sections, _ = _native_license_text_sections(inventory)
    for package, path, raw in native_sections:
        append_section(package, path, raw)
    return [(" / ".join(packages), path, raw) for packages, path, raw in sections]


def render_notices(inventory: dict[str, Any]) -> bytes:
    """Render the reviewable text notice, retaining local license bytes."""

    lines: list[str] = [
        "TreeLineWarning desktop third-party notices",
        "=============================================",
        "",
        "ENGINEERING INVENTORY ONLY: this file is not legal clearance.",
        "Verify included files, exact source provenance, license obligations, and redistribution",
        "terms before shipping. UNVERIFIED_SOURCE_PROVENANCE entries require review.",
        "",
        "Release inputs",
        "--------------",
    ]
    for key, value in inventory["release_inputs"].items():
        lines.append(f"{key}: {value}")
    lines.extend(
        [
            "",
            "Python distributions",
            "--------------------",
        ]
    )
    for distribution in inventory["python"]["distributions"]:
        if distribution.get("status") == _MISSING:
            lines.append(f"- {distribution['requested_name']}: {_MISSING}")
            continue
        expression = distribution.get("license_expression") or distribution.get(
            "metadata_license_summary"
        )
        lines.append(
            f"- {distribution['name']} {distribution['version']} "
            f"[{distribution['classification']}; license="
            f"{expression or 'metadata did not declare one'}]"
        )
        if not distribution.get("license_files"):
            lines.append(
                "  local license/notice text: NOT FOUND under this exact dist-info directory"
            )
        for item in distribution.get("license_files", []):
            lines.append(f"  license: {item['path']} sha256={item['sha256']}")
    python_license = inventory["python"]["license"]
    lines.append(
        f"- CPython {inventory['python']['version']} license: {python_license['path']} "
        f"sha256={python_license.get('sha256')}"
    )
    lines.extend(["", "Vendored browser payload", "------------------------"])
    for asset in inventory["browser"]["assets"]:
        lines.append(
            f"- {asset['path']} bytes={asset.get('bytes')} sha256={asset.get('sha256')}"
        )
    lines.append("")
    for component in inventory["browser"]["components"]:
        lines.extend(
            [
                f"{component['name']} {component['version']} ({component['package']})",
                f"  candidate license: {component['license']}",
                f"  registry metadata: {component['registry_metadata']}",
                f"  tarball: {component['tarball']}",
                f"  npm integrity: {component['integrity']}",
                f"  npm shasum: {component['shasum']}",
                f"  upstream source: {component['upstream_source']}",
                f"  status: {component['status']}",
            ]
        )
        if component.get("license_source"):
            lines.append(f"  license source: {component['license_source']}")
        for notice_line in component.get("inline_notice", []):
            lines.append(f"  inline bundle notice: {notice_line}")
        lines.append(f"  note: {component['status_reason']}")
    lines.extend(
        [
            "",
            "License corrections from earlier POC notes",
            "--------------------------------------------",
            "- laz-perf@0.0.7: official package/source metadata reports Apache-2.0;",
            "  the old LGPL-2.1 claim is superseded.",
            "- copc@0.0.8: official package metadata reports MIT; the old ISC claim is superseded.",
            "",
            "Staged native runtime",
            "---------------------",
            f"manifest: {inventory['native']['manifest']['path']} "
            f"sha256={inventory['native']['manifest'].get('sha256')}",
            f"conda byte-match audit: {inventory['native']['conda_matches'].get('path')} "
            f"status={inventory['native']['conda_matches'].get('status')} "
            f"matched={inventory['native']['conda_matches'].get('matched_count')}",
            f"versions: {json.dumps(inventory['native']['versions'], sort_keys=True)}",
            f"PROJ data: {json.dumps(inventory['native']['proj_data'], sort_keys=True)}",
            f"optional plugin policy: {inventory['native'].get('optional_plugin_policy')}",
        ]
    )
    for item in inventory["native"]["files"]:
        file_item = item.get("file", {})
        lines.append(
            f"- {item.get('path')} bytes={file_item.get('bytes')} "
            f"sha256={file_item.get('sha256')}"
        )
    lines.append("")
    for component in inventory["native"]["components"]:
        lines.extend(
            [
                f"{component['name']} observed_version={component.get('observed_version')}",
                f"  candidate license: {component['license_candidate']}",
                f"  upstream source: {component['upstream_source']}",
                f"  license source: {component['license_source']}",
                f"  status: {component['status']}",
                f"  note: {component['status_reason']}",
            ]
        )
    webview = inventory["webview2_prerequisite"]
    installer = webview["installer"]
    authenticode = webview.get("authenticode", {})
    lines.extend(
        [
            "",
            "WebView2 standalone prerequisite",
            "---------------------------------",
            f"installer: {installer['path']} bytes={installer.get('bytes')} "
            f"sha256={installer.get('sha256')}",
            f"Authenticode record: {webview['authenticode_record']['path']}",
            f"EULA record: {webview['eula_record']['path']} "
            f"sha256={webview['eula_record'].get('sha256')}",
            f"Authenticode status: {authenticode.get('status')}; "
            f"subject={authenticode.get('subject')}",
            f"installer file version: {authenticode.get('fileVersion')}; "
            f"executed={authenticode.get('executed')}",
            f"status: {webview['status']}; clean_pc_verified={webview['clean_pc_verified']}",
            "The signed offline installer is recorded as a download-only prerequisite",
            "and was not executed.",
            "The wrapper file version is not a claim about the installed WebView2 browser version.",
            "",
            "Unresolved provenance table",
            "---------------------------",
        ]
    )
    for item in inventory["unresolved_provenance"]:
        lines.append(f"- {item['status']}: {item['artifact']} — {item['reason']}")
    lines.extend(["", "Exact local license texts", "-------------------------"])
    header = "\n".join(lines) + "\n"
    output = bytearray(header.encode("utf-8"))
    for package, path, raw in _license_text_sections(inventory):
        output.extend(
            f"\n----- BEGIN LICENSE: {package} :: {path} -----\n".encode()
        )
        output.extend(raw)
        if not raw.endswith(b"\n"):
            output.extend(b"\n")
        output.extend(f"----- END LICENSE: {package} :: {path} -----\n".encode())
    eula_path = inventory["webview2_prerequisite"]["eula_record"]["path"]
    if inventory["webview2_prerequisite"].get("eula_html_present"):
        eula_file = Path(eula_path)
        if not eula_file.is_absolute():
            eula_file = REPO_ROOT / eula_file
        payload = json.loads(eula_file.read_text(encoding="utf-8"))
        output.extend(b"\n----- BEGIN MICROSOFT WEBVIEW2 EULA (UNCHANGED HTML) -----\n")
        output.extend(str(payload["evergreenHtml"]).encode("utf-8"))
        output.extend(b"\n----- END MICROSOFT WEBVIEW2 EULA (UNCHANGED HTML) -----\n")
    return bytes(output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv-root", type=Path, default=DEFAULT_VENV_ROOT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--proj-data-root", type=Path, default=DEFAULT_PROJ_DATA_ROOT)
    parser.add_argument("--viewer-root", type=Path, default=DEFAULT_VIEWER_ROOT)
    parser.add_argument("--webview-root", type=Path, default=DEFAULT_WEBVIEW_ROOT)
    parser.add_argument("--output-text", type=Path, default=DEFAULT_OUTPUT_TEXT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing generated notice pair after reviewing the inputs",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    inventory = collect_inventory(
        venv_root=args.venv_root,
        runtime_root=args.runtime_root,
        proj_data_root=args.proj_data_root,
        viewer_root=args.viewer_root,
        webview_root=args.webview_root,
    )
    if not args.overwrite and (args.output_text.exists() or args.output_json.exists()):
        raise SystemExit(
            "refusing to overwrite an existing notice output; pass --overwrite "
            "after reviewing the selected inputs"
        )
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_text.open("wb") as handle:
        handle.write(render_notices(inventory))
    with args.output_json.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(inventory, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(args.output_text)
    print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
