# P3A — Potree Viewer Proof-of-Concept (POC)

**Status: RETIRED_AS_PRIMARY_VIEWER** — Potree remains a historical/comparative
POC only.

A minimal, loopback-only Potree web viewer that renders a bounded real-RGB
sample of the protected colorized source. The POC exists to answer the design
question from `docs/project-based-multispan-viewer-design.md` §6/§17: *can
Potree 1.8.2 + PotreeConverter 2.1.3 stream a representative RGB cloud in a
local viewer before any full-scene conversion is committed?*

## Final viewer status

```
POTREE_STATUS: RETIRED_AS_PRIMARY_VIEWER
```

Reason: the Owner interaction comparison selected **Giro3D + COPC** as the
production viewer foundation (see `docs/p3a-giro3d-copc-poc.md`).  Potree is
retained as a historical/comparative POC only — its files are not deleted, and
its historical benchmark results are not rewritten.

The default acceptance dataset is the **dense span ROI** (`p3a_roi`, 982,504
points) around the validated A–B poles; the sparse whole-scene stride sample
(`p3a_sample`, 299,303 points) is kept as a technical/global/debug POC.

The protected source and the full 105,055,350-point conversion were **not**
touched or run.

---

## 快速启动 (Quick start for the Owner)

Double-click nothing — one command from the repository root:

```
cache\p3a_potree\python\p3a-venv\Scripts\python.exe scripts\serve_p3a_potree.py
```

Your default browser opens `http://127.0.0.1:8090/`. The viewer shows the
**dense 982,504-point span ROI** (real RGB) around the validated A–B poles and
frames it automatically. Navigate with:

| Action        | Mouse / touch                              |
|---------------|--------------------------------------------|
| Rotate / orbit | **left-drag** (orbit controls)             |
| Pan            | **right-drag**                             |
| Zoom           | **mouse wheel**                            |
| Fit / Reset    | top-right button「Fit / Reset View」       |
| Point budget   | top toolbar `Point Budget` slider          |
| Color mode     | `Appearance` → `Active Attribute`          |

A small hint in the bottom centre shows `左键拖动: 旋转 | 右键拖动: 平移 |
滚轮: 缩放`. Expected: a dense, colorized span scene with recognizable poles /
towers / conductors / ground context. No error banner should appear; if one
does, see [Troubleshooting](#troubleshooting).

To stop the server, close the terminal or press `Ctrl+C`.

To suppress the browser auto-open (e.g. for scripts/tests):

```
...\python.exe scripts\serve_p3a_potree.py --no-open --port 8091
```

---

## What was built

| Path | Purpose |
|------|---------|
| `scripts/prepare_p3a_potree_sample.py` | Streams the protected source **read-only**, keeps every 351st point (deterministic, spatially representative), writes `cache/p3a_potree/input/p3a_sample.las` (299,303 pts, RGB, LAS 1.4 / point format 7). Hard-capped at `MAX_P3A_SAMPLE_POINTS=1,000,000`; fails closed if the source changes while sampling. |
| `cache/p3a_potree/input/p3a_sample.las` | The bounded RGB sample (10.3 MB). |
| `cache/p3a_potree/sample-provenance.json` | Derivation provenance: source size/mtime before & after (immutability `PASS`), sampling rule, sample SHA-256. Not a scientific `SourceIdentity`. |
| `cache/p3a_potree/dataset/p3a_sample/` | PotreeConverter 2.1.3 output of the global sample: `octree.bin` + `hierarchy.bin` + `metadata.json` (octree v2.0, `position` + `rgb`, explicit `-m poisson`). |
| `cache/p3a_potree/input/p3a_roi.las` | **Owner acceptance sample**: dense real-RGB XY corridor around the validated A–B span (raw 16,702,553 pts → deterministic stride 17 → 982,504 pts, ≤ 1M cap). |
| `cache/p3a_potree/roi-provenance.json` | ROI derivation provenance: corridor geometry, raw count, stride, sample SHA-256, source immutability `PASS`. |
| `cache/p3a_potree/dataset/p3a_roi/` | PotreeConverter 2.1.3 output of the ROI sample: `octree.bin` (17.7 MB) + `hierarchy.bin` + `metadata.json` (982,504 pts, `position` + `rgb`, explicit `-m poisson`). |
| `scripts/serve_p3a_potree.py` | Strictly loopback-only `127.0.0.1` server (no `--host` override) with Windows-exclusive port binding (`SO_EXCLUSIVEADDRUSE`), exposing exactly three virtual roots: `/potree/`, `/pointclouds/`, `/`. Path traversal is rejected (verified by test). |
| `viewer_poc/p3a/index.html` | Potree 1.8.2 page (same asset set as the official `lion_las.html` example). |
| `viewer_poc/p3a/viewer.js` | Bootstrap: preflights `metadata.json`, builds `Potree.Viewer` with explicit `setControls(viewer.orbitControls)`, loads the octree, forces RGB material; frames the tight point bounds after load (not the inflated octree cube); provides a visible Fit / Reset View button and a Chinese control hint; every startup/load failure (missing metadata, JS error, unhandled rejection, load timeout) shows a visible in-page error banner. |
| `viewer_poc/p3a/viewer_perf.js` | **Debug-only** performance instrumentation + benchmark tooling: bounded rolling frame statistics (average/median/P95/interval FPS), a 4 Hz debug overlay, a fixed two-state (INTERACTIVE/IDLE) policy that is inert because the interactive and idle profiles are identical to the fixed default, a benchmark recorder with **true orbit / auto-reframe**, one-click Stage A / Final Confirmation / 400k-orbit runners, and visible-point/node interval aggregation. Normal viewer startup never auto-runs benchmarks. |
| `viewer_poc/p3a/poc_config.json` | Tuning knobs (FOV, point budget, EDL, active attribute, point size, load timeout, control help) and the authoritative active dataset URL (`pointclouds/p3a_roi/metadata.json`). The fixed normal default is **750,000 pts / ADAPTIVE / EDL ON / pixel ratio 1**. |
| `cache/p3a_potree/measurements.json` | Versions, hashes, point counts, and the full-source estimate. |
| `tests/scripts/test_prepare_p3a_potree_sample.py` | Sample logic on synthetic RGB LAS (no real source). |
| `tests/scripts/test_serve_p3a_potree.py` | Server loopback binding, asset serving, octree serving, traversal-block proof. |
| `tests/viewer/test_p3a_static_viewer.py` | Static deliverable checks; every Potree asset reference resolves on disk. |
| `tests/viewer/test_p3a_perf_policy.py` | Performance instrumentation / policy / benchmark-recorder tests (the 31 extra tests in the broader P3A performance-policy suite). |
| `docs/p3a-potree-poc.md` | This document. |

## Official dependency provenance

| Component | Version | Archive SHA-256 |
|-----------|---------|-----------------|
| Potree (viewer) | 1.8.2 | `c140a8bce0ce129de762fbc0fd3ba7c21120ffd36d423dd2755abb39f32aebe5` |
| PotreeConverter | 2.1.3 | `957279e35886c63b4aba5d6383f11f3105ee889cb5391b05307265d00cd2d775` |

Both are the official GitHub release assets; every entry passes full CRC
(`testzip`) verification. No MSVC/CMake toolchain was installed — the official
prebuilt converter runs as-is.

- **Python:** 3.11.9 (`cache/p3a_potree/python/p3a-venv`)
- **Python packages:** laspy 2.7.0, numpy 2.4.6, pytest 9.1.1
- **Active acceptance sample (ROI):** 982,504 points (raw ROI 16,702,553, stride
  17), RGB present, LAS 1.4 / format 7,
  SHA-256 `c8cf1d0fc6e7af87bc945bcf161bde242a32397adfb09a3a13a2e042ac09ab0b`
  (see `cache/p3a_potree/roi-provenance.json`)
- **Global technical sample (kept):** 299,303 points (stride 351), RGB present,
  SHA-256 `3706c2a4e3201e0bb407e50d004f03621b9681c47d33597fddfb4619712245ef`
- **Source size:** 6,303,322,159 bytes = 6.303322 decimal GB = 5.870426 GiB
  (recorded explicitly in `measurements.json`)
- **ROI conversion:** PotreeConverter 2.1.3, explicit `-m poisson --attributes "rgb"`,
  converter duration ~1.0 s, octree v2.0, 982,504 pts, 17.7 MB octree.bin

## Fixed normal performance default

The reviewed, fixed cross-device default that normal startup applies:

| Setting | Default |
|---------|---------|
| Point budget | **750,000** |
| Point size mode | **ADAPTIVE** |
| EDL | **ON** |
| Renderer pixel ratio | **1** |

Classification: **conservative cross-device default**.  Why 750k was selected:

- Measured (Stage A + Final Confirmation, RTX 5070 dev laptop at 240 Hz): 750k
  Orbit `intervalFps` 192.9 vs 1M Orbit 240.1, while 750k Orbit
  `visiblePointsAverage` 565,245.9 vs 1M 566,656.8 — only ~0.25% less visible
  detail than 1M at a lower point budget.
- 750k provides substantially more detail than 500k, approaches the effective
  visible detail of 1M in the validated ROI, stays fully responsive, and leaves
  headroom compared with 1M.
- 1M is demonstrably safe on the current development machine and yields slightly
  more visible detail, but a single machine cannot prove 1M is the best
  universal cross-device default, so 750k is retained.

The design remains intentionally simple: **no motion/idle dual budgets, no
runtime auto tuning, no GPU detection, no hardware tiers, no FIXED point mode,
no EDL disabling, no renderer `setSize` patch** (setSize is called once/frame
with zero width/height changes and is not a measured bottleneck).

The `viewer_perf.js` INTERACTIVE/IDLE policy is inert by construction — both
profiles equal the fixed 750k default — so normal startup always uses exactly
the fixed profile.  The benchmark recorder is debug tooling only and never runs
automatically on startup.

## Performance benchmark semantics

- **True orbit / auto-reframe:** each benchmark run reframes to the tight-bounds
  fit and uses a real orbit around the look-at pivot (or a lateral pan), so the
  cloud stays framed — not a camera that turns in place.
- **Protocol per run:** deterministic fit → settle → 5 s warm-up → 15 s
  measurement → auto-stop (manual Stop also works and cancels all timers).
- **`intervalFps`** = `frames × 1000 / measurementDurationMs` — the true achieved
  frame rate over the record window (not `1000/median`, which over-reports when
  frames are dropped).  The median-derived value is kept separately as
  `medianFps`.
- **Visible aggregates:** during the measurement window the recorder samples
  `pointcloud.numVisiblePoints` / `numVisibleNodes` each frame (Potree already
  computes them; no octree traversal) into a bounded buffer and exports
  `visiblePointsAverage/Median/P95/Max`, `visibleNodesAverage/Median/Max`, with
  `actualVisiblePointsAtEnd` as a secondary diagnostic.
- **Result storage** is capped (`MAX_BENCHMARK_RESULTS`, FIFO drop-oldest) so
  completed snapshots cannot grow without bound.

## CloudCompare reference

The performance direction (loaded vs per-frame rendered point counts, octree/LOD
visibility selection, fixed point-count-limited LOD work, reduced rendering
complexity during motion) was informed by studying the official **CloudCompare**
source as a conceptual reference only.  **No CloudCompare code was copied,
translated, or reused** (`CLOUDCOMPARE_CODE_REUSED: NO`).

## Full-source estimate (non-authoritative)

The dense ROI (982,504 pts in a ~75 × 50 m corridor) confirms local density far
exceeds the whole-scene average; a full 105,055,350-point conversion is
**intentionally not** run for the POC (estimate basis in `measurements.json`).

## Rebuilding (only if needed)

```
# regenerate the global technical sample (read-only on source) + sample-provenance.json
cache\p3a_potree\python\p3a-venv\Scripts\python.exe scripts\prepare_p3a_potree_sample.py

# regenerate the dense ROI sample (read-only on source) + roi-provenance.json
cache\p3a_potree\python\p3a-venv\Scripts\python.exe scripts\prepare_p3a_potree_sample.py ^
  "--pole-a=-66.3449 2.2295 0.8314" "--pole-b=-25.3627 20.4183 -0.2397" ^
  --target-points 1000000 --output cache\p3a_potree\input\p3a_roi.las ^
  --provenance cache\p3a_potree\roi-provenance.json

# reconvert (explicit poisson + rgb)
cache\p3a_potree\tools\PotreeConverter_2.1.3_x64_windows\PotreeConverter.exe ^
  -i cache\p3a_potree\input\p3a_roi.las -o cache\p3a_potree\dataset\p3a_roi -m poisson --attributes "rgb"
```

## Automated tests

Run from the repository root with the temp-root alignment required by the
Windows test harness (see `tests/` conventions):

```
set PYTEST_DEBUG_TEMPROOT=E:\TreeLineWarning-fast\.pytest_tmp\ltmp
set TEMP=%PYTEST_DEBUG_TEMPROOT%
set TMP=%PYTEST_DEBUG_TEMPROOT%
cache\p3a_potree\python\p3a-venv\Scripts\python.exe -m pytest tests/scripts/test_prepare_p3a_potree_sample.py tests/scripts/test_serve_p3a_potree.py tests/viewer/
```

Two distinct results, so they are not confused:

- **Final three-file closure suite** (the exact closure-verification scope):
  `tests/scripts/test_prepare_p3a_potree_sample.py`,
  `tests/scripts/test_serve_p3a_potree.py`,
  `tests/viewer/test_p3a_static_viewer.py` → **60 passed**.
- **Broader P3A performance-policy suite**: the three files above **plus**
  `tests/viewer/test_p3a_perf_policy.py` (31 tests) → **91 passed**.

Neither number is a substitute for the other.  Historical: before the
performance-policy work, the three-file focused count was **56 passed** (that
earlier count is retained here only as clearly-labeled history).

The health endpoint `/p3a_healthz` verifies the **active configured dataset**
(`p3a_roi`) from `poc_config.json` — it never reports ready because an unrelated
old sample exists.

## Troubleshooting

- **Blank page / no points:** open `http://127.0.0.1:8090/p3a_healthz` — it
  must return `{"status": "ok", "dataset_ready": true, ...}`. If
  `dataset_ready` is false, run the rebuild steps above.
- **Port 8090 busy:** start with `--port 8091` and open that URL.
- **Cloud appears tiny / off-centre:** click the top-right「Fit / Reset View」
  button, which frames the tight point bounds of the active ROI.
- **Rotation feels wrong / pan-only:** the viewer explicitly selects Potree 1.8.2
  `orbitControls` (`viewer.setControls(viewer.orbitControls)`): left-drag
  rotates, right-drag pans, wheel zooms. If only one button responds, check that
  no browser extension or second window is intercepting the canvas.
- **WebGL black screen:** confirm the browser allows hardware WebGL/WebView2;
  toggle `viewer.setEDLEnabled(false)` in `viewer.js` or the config if the GPU
  driver fights EDL. This is exactly the compatibility risk the POC exists to
  measure (design §6).
- **Never use `file://` to open `index.html`** — the viewer needs the HTTP
  server for `fetch` and worker loading.

## Protection guarantees

- The protected source `input_las/2026-08-07_16没54-10_PointCloud_colorized.las`
  is streamed **read-only**; it is unmodified (measurements.json records it).
- The server rejects any path that escapes its three virtual roots; the test
  `test_traversal_cannot_reach_protected_source` proves the 105M-point source
  is unreachable over HTTP.
- No full 105M conversion was performed.
