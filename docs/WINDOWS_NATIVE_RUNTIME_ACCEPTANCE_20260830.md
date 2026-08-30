# Windows native runtime engineering acceptance — 2026-08-30

This records source/staged-runtime checks, not frozen EXE or clean-PC acceptance.
No frozen science, goldens, REAL project membership, or raw LAS was modified.

## Release inputs

- `.build-pdal-runtime-final-20260830`: 71 real files, materialized from the
  approved historical rc-20260823f package; no junctions or optional PDAL plugins.
  Manifest SHA-256:
  `db5192698a761acc382c4deed9319e3251a83dbc14d83bbf933e59e77295d353`.
- PDAL 2.10.2, Python binding 3.5.4 / CPython 3.11 x64, GDAL 3.13.1,
  native PROJ 9.8.1; lazrs 0.8.2 supplies laspy LAZ support.
- `.build-pdal-proj-data-final-20260830-02`: 16 PROJ data files plus seven
  provenance files. Manifest SHA-256:
  `b60944ecc06c5f1cd60c74dd0c0c04a0e8882dc93aabad89add0c2ffb935acb7`.
  The data and native PROJ DLL match the locally cached conda-forge package
  `proj-9.8.1-hd30e2cd_0`; its archive, metadata, payload hashes, DLL and COPYING
  are recorded. Native database layout is major 1 / minor 6.
- pyproj retains its own wheel database. Native PDAL selects the separate,
  compatible PROJ data directory; network grid retrieval is disabled.

## Repairs and independent checks

The older pyproj database could not satisfy native PROJ 9.8.1. The separate
source-bound overlay fixes native EPSG:4326 → EPSG:3857 at 120°,30° without
replacing pyproj's database. The observed output is approximately
13,358,338.8952 m / 3,503,549.8435 m.

An additional Windows DLL collision was reproduced: loading Python's older
SQLite first prevented PDAL import. The runtime now preloads the approved native
SQLite DLL, retaining its handle. The build uses isolated database inspection,
preloads before native collectors, and rewrites every collected `sqlite3.dll`
source to the same approved bytes while retaining each destination.

The supervisor independently checked runtime configuration → Python sqlite3
3.53.3 → an in-memory transaction → PDAL import and complete dependency health.
The health check includes Pillow PNG, SMRF/Ground/HAG, native EPSG transformation,
pyproj offline transformation, and LAZ → PDAL LAS values. A separate AST-isolated
check verified that both root and nested SQLite TOC entries select approved bytes.

Evidence under `acceptance_evidence/final_delivery_20260830/`:

- `sol_native_sqlite_import_order.json`: original DLL-order failure preserved.
- `sol_native_sqlite_repaired_health.json`: independent operation and TOC checks.
- `sol_native_sqlite_closed_tests.log`: one fixture failure / 17 passes preserved;
  the fixture accidentally retained five points while expecting one.
- `sol_native_sqlite_final_tests.log`: corrected fixture, **18 passed in 8.33 s**.
  Tests cover build contracts, runtime health, staged packaging and PROJ materialization.

Native binary provenance was independently matched against cached conda package
records: 68/71 files match, including all native binaries; three generated Python
bytecode files have no package byte match. The notice inventory records remaining
provenance and redistribution review items. These engineering checks do not grant
legal clearance for external distribution.

## Remaining acceptance

The actual PyInstaller output must independently pass sanitized-environment
dependency health, read-only REAL report/restart, relocated-path and native-window
checks. A clean PC without developer runtimes has not been tested. Do not infer
either acceptance from the staged checks above.
