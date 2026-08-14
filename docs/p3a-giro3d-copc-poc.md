# P3A — Giro3D + COPC Viewer (Selected)

**Status: VIEWER_GATE — PASS · SELECTED_VIEWER: Giro3D + COPC (final decision)**

A viewer candidate for the P3A comparison (CloudCompare vs Potree vs Giro3D).
It renders the **same dense real-RGB ROI** (982,504 points) that Potree renders,
from a derived **Cloud-Optimized Point Cloud (COPC)** file using the Giro3D
engine.

## Final Viewer Gate decision

```
VIEWER_GATE: PASS
SELECTED_VIEWER: Giro3D
VIEWER_DATA_FORMAT: COPC
PRODUCTION_COPC_WRITER: PDAL writers.copc
CUSTOM_COPC_BUILDER: POC_ONLY (NOT_PRODUCTION)
POTREE_STATUS: RETIRED_AS_PRIMARY_VIEWER
SCIENTIFIC_COMPUTATION_SOURCE: FULL_RESOLUTION_RAW
RAW_WIRE_VIEW_COMPLETENESS_REQUIRED: NO
RAW_GROUND_VIEW_COMPLETENESS_REQUIRED: NO
CLICK_XYZ: PASS
FULL_SCALE_OWNER_VALIDATION: PASS
```

**Viewer role** (closed): scene navigation · tower review · approximate XYZ
picking · visualization of scientific result overlays.

**Scientific boundary — IMPORTANT:** viewer-displayed COPC/LOD points MUST NOT
be authoritative inputs for: ground, HAG, wire reconstruction, vegetation
membership, tree segmentation, TreeMetrics, or clearance computation.  All
scientific computation uses the full-resolution raw source
(`SCIENTIFIC_COMPUTATION_SOURCE: FULL_RESOLUTION_RAW`), never the viewer's
decimated COPC representation.

## Next phase

```
NEXT_PHASE: V5-A Tower Candidate Detection + Human Review
```

The Viewer Gate is closed after Git closure of this branch.  V5-A uses the
full-resolution raw scientific source (not the viewer COPC) for tower candidate
detection and human review.

Purpose is **not** to replace Potree yet — it is to let the Owner directly
compare interaction and visual quality on identical data.

## Quick start

```
# 1. Convert the ROI LAS to COPC (done once; output is an ignored cache):
cache\p3a_giro3d\python\giro3d-venv\Scripts\python.exe scripts\prepare_p3a_giro3d_copc.py --validate

# 2. Serve the viewer (loopback-only, default port 8091):
cache\p3a_potree\python\p3a-venv\Scripts\python.exe scripts\serve_p3a_giro3d.py
```

Then open `http://127.0.0.1:8091/`.

| Interaction | Control |
|-------------|---------|
| Rotate / orbit | **left-drag** |
| Pan | **right-drag** |
| Zoom | **mouse wheel** |
| Fit / Reset | top-left button |
| Click -> world XYZ | **single click** on the cloud |

A small diagnostics box (top-right) shows engine / dataset / point count /
loading / camera position / last clicked XYZ.

## Data

- **Input (shared with Potree):** `cache/p3a_potree/input/p3a_roi.las` —
  982,504 points, real RGB, LAS 1.4 / point format 7.  Never modified.
- **Output COPC:** `cache/p3a_giro3d/dataset/p3a_roi.copc.laz` (4.85 MB, 21
  octree nodes, depth 0..3, 982,504 points).
- **Provenance:** `cache/p3a_giro3d/copc-provenance.json` (input path + count,
  output SHA-256, tool, version, license, bounds).
- The protected source and the protected authoritative run are never touched.

### COPC conversion tool

| Field | Value |
|-------|-------|
| Tool | `copclib` (Python bindings of the **copc-lib** reference library) |
| Version | copc-lib 2.6.3 |
| LAZ compression | bundled **lazperf** (LASzip-compatible) |
| License | copc-lib **BSD-3-Clause**; lazperf (LASzip-compatible) |
| Octree | capacity-based spatial decimation with Morton-order spread (root
  holds a well-spread sample; every point lives in exactly one node; natural LOD) |

The conversion is **derived / reproducible** — run
`scripts/prepare_p3a_giro3d_copc.py` to regenerate.

## Dependency / license record

| Component | Version | License |
|-----------|---------|---------|
| Giro3D | 2.0.3 | MIT |
| Three.js | 0.180.0 | MIT |
| OpenLayers | 10.10.0 | BSD-2-Clause |
| proj4 | 2.21.0 | MIT |
| copc.js (COPC reader) | 0.0.8 | ISC |
| laz-perf (LAZ decompression WASM) | 0.0.7 | LGPL-2.1 (with the WASM used for
  decoding; the viewer serves it locally, never from a CDN) |
| copclib (COPC writer) | 2.6.3 | BSD-3-Clause |

All are compatible with an internal research POC.  No incompatible code is used
silently; the exact versions and licenses are recorded here.

The Giro3D bundle is built from source with esbuild and vendored under
`viewer_poc/giro3d/vendor/giro3d.bundle.js`; the laz-perf WASM is vendored under
`viewer_poc/giro3d/vendor/laz-perf/`.

## What is implemented

- RGB point cloud rendering (Giro3D COPC source, laz-perf LAZ decompression in
  a web worker).
- Orbit (left-drag), pan (right-drag), zoom (wheel) via Three.js `MapControls`
  with damping.
- **Useful initial framing** from the **occupied dataset bounds** (read from the
  COPC LAS header), not the protected pole coordinates — the corridor/tower
  structure is recognizable immediately.
- Fit / Reset button (reframes the occupied bounds).
- **Click -> world XYZ**: point picking returns the source/world coordinate
  frame (the ROI's local metric frame; no projection, so XYZ are the same
  numbers as the source).
- Loading status + visible error state (fail-closed banner).
- Small diagnostics section (engine / dataset / point count / loading / camera
  / last clicked XYZ).
- Loopback-only server with exclusive Windows bind, safe virtual roots, HTTP
  Range support (required by copc.js), and a health endpoint reporting the
  active COPC.

Not implemented: production UI, tower review UI, prediction, point picking
beyond the XYZ read-out, projection workflows, any scientific pipeline.

## Tests

```
set PYTEST_DEBUG_TEMPROOT=E:\TreeLineWarning-fast\.pytest_tmp\ltmp
set TEMP=%PYTEST_DEBUG_TEMPROOT%
set TMP=%PYTEST_DEBUG_TEMPROOT%
cache\p3a_potree\python\p3a-venv\Scripts\python.exe -m pytest tests/viewer/test_p3a_giro3d_static.py tests/scripts/test_serve_p3a_giro3d.py
```

Covers: required static files, configuration matching the ROI, active-COPC
header point count, pure decision helpers (bounds parse / framing / click XYZ)
via Node, loopback-only + exclusive-bind server, safe file roots, Range serving,
health endpoint not-ready-when-COPC-missing, and no automatic scientific
pipeline in the viewer.

## Full-source experimental conversion (105,055,350 points — historical)

A **historical full-scale experimental conversion** of the complete protected
source DID occur and was Owner-tested on the full 105,055,350-point scene
(smooth; full scene loads).  It is NOT the production cache:

- **Experimental COPC:** `cache/giro3d_full_source/dataset/p3a_full.copc.laz`
  (536.9 MB, 3,516 octree nodes, depth 0–9, 105,055,350 points), provenance in
  `cache/giro3d_full_source/copc-provenance.json`.
- **Experimental builder:** `scripts/prepare_p3a_giro3d_full_copc.py` (streams
  the protected source read-only into memmap, builds the octree with bounded
  memory, writes the COPC; round-trip verified all 105M points; source
  size/mtime/SHA-256 recorded and unchanged).  Transient memmaps are deleted.
- **Why it is POC-only:** the trusted PDAL 2.10.2 `writers.copc` control showed
  the custom builder's COPC root `spacing` semantics are incorrect
  (span-based: 85.78 m ROI / 310.65 m full) versus the standard density-derived
  value PDAL writes (0.5835 m ROI), which corrupts Giro3D's geometricError /
  LOD refinement.  **Production COPC generation must use `PDAL writers.copc`.**
- The experimental full COPC is kept only as historical evidence; it is NOT the
  production viewer cache.
- The ROI POC viewer (`8091` / A/B `8093`-custom `8094`-PDAL) is untouched.

## Protection guarantees

- The protected source and protected authoritative run are untouched.
- The ROI is shared (not regenerated).
- The server rejects any path escaping its virtual roots; the protected source
  is never reachable over HTTP.
- A full derived COPC of the 105M source was created ONCE as an experiment for
  the Viewer Gate (see above); it is POC-only and is not a production cache.
  No protected data was modified.
