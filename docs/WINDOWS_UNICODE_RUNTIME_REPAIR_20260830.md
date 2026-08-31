# Native PROJ/GDAL Unicode path repair — 2026-08-30–31

The first Survey baseline passed native health, REAL current-report/restart and
GUI checks at its original path. A byte-identical copy under
`dist/交付验收 含空格/TreeLineWarning Survey 基线01` failed native PDAL's EPSG
lookup with `Cannot find proj.db`. The database was present and its hash matched
the approved PROJ 9.8.1 data. Pyproj's independent wheel database still worked.
The failed run stopped before opening the REAL project, and its preservation
guard passed. This failure is retained in
`acceptance_evidence/final_delivery_20260830/packaged_acceptance_baseline01_relocated`.

An isolated one-point probe distinguished the Windows narrow C environment
from missing or incompatible data: the ordinary Unicode environment path failed,
whereas an explicitly UTF-8 native search path allowed both PROJ EPSG creation
and PDAL reprojection. Changing the narrow environment to UTF-8 was investigated
but is not the repair because it can make the Windows wide environment disagree.

The initial default-context repair was insufficient: GDAL creates its own
thread-local PROJ contexts and can replace inherited paths with the narrow
environment value. The resumed investigation reproduced this failure rather than
treating the earlier isolated probe as complete acceptance.

The engineering repair calls `proj_context_set_search_paths` on the exact bundled
`proj_9.dll` and `OSRSetPROJSearchPaths` on the exact bundled `gdal.dll`, supplying
the existing bundled data path encoded as UTF-8 before PDAL imports. Both DLL
handles are retained and both APIs explicitly disable network access.
PROJ documents that contexts created later inherit the default-context search
paths and that those paths use UTF-8 in [its C API reference](https://proj.org/en/stable/development/reference/functions.html#c.proj_context_set_search_paths).
GDAL exposes its separate search-path and network settings in [the spatial
reference C API](https://gdal.org/en/stable/api/ogr_srs_api.html).
No ASCII alias, short path, code-page change, resource relocation, database
replacement, or science change is part of this repair.

The runtime treats its native library/data selection as fixed for a process.
Repeated initialization of the same selection is allowed; switching a configured
selection must fail explicitly because already-created native contexts may retain
the earlier data. Missing native data, DLLs or required APIs fail closed.

The earlier package and relocated copy are preserved. This source repair requires
a new build and independently tagged frozen acceptance; source tests alone do not
close the package gate. Clean-PC installation and third-party legal clearance
remain separate, unverified delivery boundaries.

## Source validation on 2026-08-31

The Sol worker's focused run passed 17 tests with one expected skip for PDAL not
being installed directly in the lean development venv. Three fresh native child
processes passed using the approved staged runtime. Both Unicode staged-data and
frozen-layout cases checked LAS and LAZ transformation, main/thread-created PDAL
contexts, native DLL origins, preserved Windows wide environment, network-off
state, and unchanged pyproj database path/hash. Evidence is in
`acceptance_evidence/final_delivery_20260830/unicode_proj_runtime_resume_20260831_01`.

The parent independently reviewed the C API signatures, fixed process selection,
failure boundaries and scientific scope, then ran the integrated runtime,
packaging, launcher diagnostics, desktop shell, build-contract and PROJ
materializer checks: **71 passed, 1 skipped in 19.69 seconds**. Source hashes were
unchanged across the run. Scoped Ruff and `git diff --check` passed. The complete
parent receipt and logs are in
`acceptance_evidence/final_delivery_20260830/sol_unicode_runtime_review_20260831`.
