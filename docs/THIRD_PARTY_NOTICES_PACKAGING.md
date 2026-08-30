# Desktop third-party notice assembly

The release package needs a reviewable notice inventory for the Python
environment, the staged PDAL DLL closure, the vendored Giro3D viewer, and the
WebView2 prerequisite.  The collector is deliberately evidence oriented.  It
does not decide whether an artifact may be redistributed and it does not turn
an upstream URL into source provenance for a historical binary.

Run the collector from the repository with the same interpreter used for the
desktop build:

```powershell
.venv311\Scripts\python.exe scripts\collect_desktop_notices.py
```

For the accepted native snapshot, write an exclusive, tagged pair so an
earlier acceptance pair remains intact:

```powershell
.venv311\Scripts\python.exe scripts\collect_desktop_notices.py `
  --proj-data-root .build-pdal-proj-data-final-20260830-02 `
  --output-text acceptance_evidence\final_delivery_20260830\THIRD_PARTY_NOTICES.final_native03.txt `
  --output-json acceptance_evidence\final_delivery_20260830\THIRD_PARTY_NOTICES.final_native03.json
```

The collector refuses to replace an existing output unless `--overwrite` is
given explicitly.

It writes:

- `acceptance_evidence/final_delivery_20260830/THIRD_PARTY_NOTICES.txt`, a
  human review copy containing the exact bytes of local `*.dist-info` license
  files (including recursive PEP 639 `licenses/` trees), the CPython license,
  the native file manifest, browser artifact hashes and source records, and
  the unchanged WebView2 EULA HTML;
- `acceptance_evidence/final_delivery_20260830/THIRD_PARTY_NOTICES.json`, a
  machine-readable inventory with package versions, local license paths and
  hashes, native payload rows, browser registry integrity values, and an
  `unresolved_provenance` table.

For native conda byte matches, the collector also reads the exact cached
package license or notice files recorded by the audit.  It validates each
file's path, byte count, and SHA-256 before embedding its unchanged bytes in
the text notice.  Identical path-and-hash pairs are emitted once while the
section header retains every matched conda package identity.  A missing,
invalid, or changed cached text is reported in `unresolved_provenance` and is
never replaced with an invented license.

The selected runtime distributions include the desktop application's direct
dependencies and the required pywebview/pythonnet closure.  PyInstaller and
its build dependencies are recorded separately as `build_tool`; this keeps the
build environment auditable without implying that every build tool file is in
the installed application.

The vendored JavaScript bundle has no lockfile or generated dependency
manifest beside it.  Giro3D 2.0.3, Three.js 0.180.0, OpenLayers 10.10.0,
proj4 2.21.0, `copc` 0.0.8, and `laz-perf` 0.0.7 are therefore recorded with
the exact npm registry integrity/shasum values found during the audit, while
the byte-to-tarball mapping remains `UNVERIFIED_SOURCE_PROVENANCE` until the
bundle can be reproduced from a locked source tree.  The official metadata
identifies `copc@0.0.8` as MIT and `laz-perf@0.0.7` as Apache-2.0; these replace
the older ISC/LGPL-2.1 claims in the POC note.

The staged PDAL files have exact local hashes and observed versions where the
PDAL probe or PE version resource supplied one.  The bounded local conda audit
now verifies 68 of 71 staged files against cached conda package bytes and
repodata hashes, including the PDAL closure and compatibility DLLs.  Those
rows are marked `LOCAL_CACHED_CONDA_PACKAGE_BYTE_MATCH`; this is an engineering
source match and does not decide license or redistribution clearance.  The
three generated PDAL `__pycache__` files remain unmatched and are listed as
`UNVERIFIED_SOURCE_PROVENANCE`.

The accepted native conda audit currently has local license or notice text for
46 matched packages.  `libsqlite-3.53.3-hf5d6505_0` has no cached local text;
its metadata declares `blessing`, referring to SQLite's public-domain
declaration.  That metadata observation is recorded as an unresolved local
text gap and is not legal clearance for redistribution.

The accepted native data snapshot is
`.build-pdal-proj-data-final-20260830-02`.  It was copied from the local
conda-forge PROJ 9.8.1 package (`hd30e2cd_0`), with every payload file checked
against `info/paths.json`; its database has layout minor 6.  The snapshot
manifest retains the package archive, `info/index.json`, `info/paths.json`,
repodata, `COPYING`, and source DLL hash proof.  The pyproj 3.7.2 wheel keeps
its own database (layout minor 4); the runtime sets native PDAL `PROJ_DATA` to
the snapshot and leaves pyproj's data directory unchanged.  Fresh native
health checks pass EPSG 4326 to 3857 at 120°,30°, alongside Ground/HAG and LAZ
round trips.

The WebView2 entry points to the Authenticode-verified, download-only
standalone x64 installer under `offline_verified/`.  It records
`executed=false`, includes the official EULA unchanged in the text notice,
and does not confuse the installer's wrapper version with the browser runtime
version.  The installer was not tested on a clean PC; retain that as a release
acceptance item.

Before creating a package, a reviewer should confirm that the PyInstaller TOC
matches the inventory, copy `THIRD_PARTY_NOTICES.txt` beside the application,
retain the WebView2 EULA and prerequisite records, and resolve every
`MISSING_LOCAL_ARTIFACT` or native/browser provenance entry that matters to the
chosen redistribution model.  Re-run the collector after any runtime,
viewer, or native snapshot change and review the changed hashes.
