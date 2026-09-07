# FULL_PIPELINE_PERFORMANCE_V1 — Gate 2 profile and GPU prototype

Date: 2026-09-07 (Asia/Shanghai)

## Gate result

`FULL_PIPELINE_PROFILE=PASS`

`EDITOR_PROFILE=PASS`

`GPU_CLEARANCE_PROTOTYPE=PASS`

`GPU_CPU_SCIENTIFIC_EQUIVALENCE=PASS`

`PRODUCTION_GPU_INTEGRATION=BLOCKED`

`PRODUCTION_BLOCKER=CONSERVATIVE_GPU_CANDIDATE_CERTIFICATE_NOT_IMPLEMENTED`

Gate 2 is complete for profiling and prototyping. The GPU implementation is deliberately not
connected to production clearance. Gate 3 CPU/GPU overlap and Gate 4 multi-span scheduling were
not started.

## Audited baseline

- Branch: `codex/final-deliverable-20260830`
- Starting HEAD: `94e411347ca4e3f2e8fb5d654e9a1fa91de1b06e`
- The reported older DBH commit `c29d3094a39f3c085bb607a51cbec52fb87e9ca3` was not HEAD.
- The working tree already contained ten modified tracked files and extensive untracked build,
  acceptance, workspace, and artifact content. Those prior changes were not overwritten.
- The Gate 2 tracked change added only the standalone GPU benchmark, its focused tests, and this
  report. Profiling run artifacts remain untracked. Gate 2 did not change Ground, HAG, Wire, Tree,
  DBH, clearance, CONTACT/DANGER, or risk policy production code.
- No EXE was built and nothing was pushed.

## Workload and measurement rules

The real workload was:

- Source: `input_las/2026-08-07_16没54-10_PointCloud_colorized.las`
- Source size: 6,303,322,159 bytes
- Source SHA-256: `6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596`
- Source points: 105,055,350
- Project: `workspace/survey_v01_gate_b_20260830`
- Topology: two spans
- Current profile run: `analysis-gate2-full-profile-20260907`
- Current wall run: `analysis-gate2-wall-20260907`
- Current reconciled workload: 17 trees and 3,766,354 tree members
- Minimum clearance: 4.543721626531224 m

Wall-clock values use `perf_counter` around the actual operation. The 83.021 s multi-span number
is the uninstrumented current wall baseline. The 105.890 s cProfile run is used only to attribute
cost because profiler overhead is material. The 421.867 s cold baseline is historical evidence
from the same 105,055,350-point source and is labelled as such; the 6.3 GB cold caches were not
destroyed and rebuilt during this audit.

## Required timing fields

`PROJECT_OPEN_SECONDS=0.010430`

Backend construction plus project and latest-batch projection only. Final-result hydration is
1.927471 s; a cold application restart through visible WebGL output is reported separately.

`POINT_CLOUD_LOAD_SECONDS=20.744295`

Effective cold import-to-overview cost derived from separately measured consecutive components:
6.024072 s source SHA-256 validation plus 14.720223 s first 90,000-point overview build. The
overview alone is 14.720223 s cold and 0.094403 s hot from cache.

`SINGLE_SPAN_TOTAL_SECONDS=40.912211`

Current real S002 wall measurement (`analysis-gate1-profile-20260907-s002`).

`MULTISPAN_TOTAL_SECONDS=83.020718`

Current real two-span uninstrumented wall measurement (`analysis-gate2-wall-20260907`). The same
workload under cProfile took 105.890245 s. The same source's historical cold multi-span baseline
was 421.867190 s.

`VIEWER_INITIAL_LOAD_SECONDS=6.739000`

Cold new backend process through WebGL2 with 50,000 tree points visible. Warm same-process reload
was 0.471000 s.

`TREE_SELECTION_SECONDS=0.010900`

Browser-published click metric for tree T-1-002. Backend-only lookup was 0.000089 s.

`SMALL_LASSO_SECONDS=0.032519`

One 106,330-point tree, selecting 11,359 points. Candidate lookup was 0.027349 s and polygon
geometry 0.001048 s; no raw LAS read occurred.

`LARGE_LASSO_SECONDS=2.130568`

All current trees, 3,750,317 projected points and 3,171,458 selected points. Candidate lookup was
0.351038 s and polygon geometry 0.278723 s; no raw LAS read occurred.

`DELETE_SECONDS=5.933420`

Worst measured delete was point deletion: 0.059295 s membership mutation, 5.870176 s full viewer
artifact/display identity rebuild, 0.003285 s journal write. Tree deletion was 4.505233 s.

`UNDO_SECONDS=4.760762`

Worst measured undo was tree-delete undo, with 4.755714 s in viewer rebuild. Point-delete undo was
4.730513 s.

`POST_EDIT_REFRESH_SECONDS=0.250943`

Worst measured post-commit refresh for a 60,000-point display. Tree-delete refresh was 0.092102 s.

`EXPORT_SECONDS=4.001295`

Real finalized 16-tree, 3,750,317-member Excel generation to an isolated profiling filename. The
temporary workbook was deleted. Hydration plus `final_reconciled()` before export was an additional
6.482944 s and is not included in `EXPORT_SECONDS`.

## Full operation table

Expected impact values are targets or bounded estimates, not achieved results, unless marked
measured.

| OPERATION | CURRENT_SECONDS | BOTTLENECK_CLASS | PROPOSED_OPTIMIZATION | EXPECTED_IMPACT | RISK |
|---|---:|---|---|---|---|
| Project open, backend projection | 0.010430 | Metadata | Keep lazy artifact hydration | Already negligible | Low |
| Final result hydration after restart | 1.927471 | CPU / full-array unique | Store and validate a directly loadable final index; avoid rebuilding identity with `np.unique` | Target below 0.5 s | Medium: persistent identity |
| Point cloud cold load to overview | 20.744295 | IO / CPU / copies | One source fingerprint per immutable import; fused chunk decode/RGB/overview sampling | Save about 6 s repeated hash plus reduce 14.7 s decode | Medium: source binding |
| Point cloud hot overview | 0.094403 | Serialization / copies | Preserve cache and avoid list materialization where possible | Target below 0.05 s | Low |
| Historical cold multi-span total | 421.867190 | IO / preprocess / CPU | Consolidate source passes and reuse verified role indexes | Largest end-to-end opportunity, over 200 s plausible | High: role membership |
| Current single-span total | 40.912211 | Mixed | Apply reconciliation/hash/clearance improvements in measured order | Must rebenchmark after each gate | Medium |
| Current multi-span total | 83.020718 | Mixed | Reconciliation, validation/hash, then certified GPU clearance | GPU alone cannot produce 2x; combined work required | Medium-high |
| Historical cold role preparation | 291.893728 | IO / CPU / memory | One source traversal feeding bounded role builders; persist content-addressed completion | Largest cold-run reduction | High: role equivalence |
| Historical source role preprocessing | 65.822421 | IO / CPU | Fuse decode and role routing, avoid full materialization | Tens of seconds | High: exact membership |
| Historical vegetation role tile index | 165.913984 | IO / sorting / copies | Streamed tile build, bounded external sort, reuse validated index | Over 100 s possible on cold path | High: vegetation membership |
| Ground HAG kernel | 1.666572 | CPU | No priority until full ground workset timing is instrumented | Small relative gain | High: frozen Ground/HAG |
| Wire role query, both spans | 0.061128 | IO / indexed query | Keep current indexed path | Negligible | Low |
| Wire science, both spans | 0.530206 | CPU | No Gate 2 change | Negligible | High: frozen Wire |
| Vegetation role query, both spans | 1.362486 | IO / indexed query | Batch adjacent span queries and reuse mapped tiles | Below 1 s target | Medium: membership |
| Tree science, both spans | 8.276512 | CPU / allocation | Profile per kernel; vectorize only proven hot loops and preserve exact membership | Several seconds possible | High: frozen tree membership |
| Per-span clearance, both spans | 4.460402 | CPU distance | Certified GPU candidate screen plus CPU exact final judge | Core kernel measured 30.07x; stage gain smaller | High: exact clearance |
| Global reconciliation | 20.512594 | Python / repeated unique | Build one global index/grouping and remove per-point/per-tree repeated `np.unique` | Target below 5 s | Very high: persistent identity/order |
| Final global clearance publish | 7.267618 | CPU distance | Same certified GPU candidate screen, CPU exact final judge and fallback | Isolated core 13.498 s to 0.449 s measured | High: exact clearance |
| Repeated cache accept/publish hashing | 12.553 cProfile | IO / SHA-256 | Memoize validated immutable artifact fingerprints in the run manifest | Roughly 10 s on repeated runs | Medium: stale-cache safety |
| Role-index load and validation | 8.681509 | IO / SHA-256 | Validate content-addressed completion once per process/run | Several seconds | Medium: stale-cache safety |
| Viewer artifact build | 1.844682 | CPU / copies | Incremental display identity and packed typed-array payload | Target below 0.5 s | Medium |
| Viewer cold initial load | 6.739000 | backend / JSON / WebGL upload | Binary/typed payload, progressive first frame, retain buffers | Target under 2 s first useful frame | Medium |
| Tree selection | 0.010900 | Viewer | Keep ID lookup; no optimization priority | Already interactive | Low |
| Small lasso | 0.032519 | Editor index | Keep scoped candidate lookup | Already interactive | Low |
| Large lasso | 2.130568 | CPU / copies / projection | Spatial screen before projection; reuse projected buffers | Target under 0.5 s | Medium: selection exactness |
| Point delete commit | 5.933420 | Full viewer rebuild | Delta membership/display update and append-only inverse operation | Target under 0.5 s | High: undo/identity |
| Tree delete commit | 4.505233 | Full viewer rebuild | Remove one tree buffer/record by stable identity | Target under 0.3 s | High: identity |
| Merge commit | 4.669295 | Full viewer rebuild | Rebuild only merged identities and affected display buffers | Target under 0.5 s | Very high: authoritative membership |
| Undo | 4.760762 | Full viewer rebuild | Apply stored inverse delta; rebuild only affected tree buffers | Target under 0.5 s | High: deterministic replay |
| Post-edit refresh | 0.250943 | Full payload fetch / setData | Versioned delta response and buffer sub-update | Target under 0.1 s | Medium: client consistency |
| Prediction export | 4.001295 | Metrics / workbook | Cache revision-bound metrics; stream workbook rows | Target below 1.5 s | High: DBH/metric contracts |

## cProfile attribution

The profiled multi-span run took 105.890245 s. Important cumulative results were:

- `reconcile_trees`: 39.423 s; `_global_reconcile`: 37.271 s.
- 3,771,286 calls to `np.unique`: 33.072 s cumulative; `_unique_hash`: 19.262 s.
- Role-tile loading/validation: 21.596 s cumulative.
- Cache accept/publish fingerprint work: 12.553 s; source fingerprinting: 12.420 s.
- Role-tile file SHA-256: 11.787 s.
- Clearance: 12.714 s cumulative; `_candidate_exact_nearest_edges`: 10.939 s.
- `_in_core_of_span`: 4.666 s and `searchsorted`: 4.698 s.

These numbers explain the code paths; they are not substituted for uninstrumented wall time.

## GPU clearance prototype

Hardware was detected at runtime, not encoded in the implementation:

- Device: NVIDIA GeForce RTX 5070 Laptop GPU
- Detected memory at benchmark start: 7,338,983,424 bytes free of 8,546,484,224 bytes
- Driver/runtime: driver 13.1-compatible, CuPy CUDA runtime 13.2
- CuPy 14.2.0 was installed only in an isolated temporary benchmark environment. No production
  dependency or package configuration was changed.
- Chunk size was calculated from detected free memory and workload size; no VRAM amount or GPU
  model is hard-coded.
- `AUTO` safely falls back to the current authoritative CPU implementation. Explicit `GPU` fails
  closed when the CUDA capability probe fails.

The real isolated workload contained 3,766,354 points and 72 canonical wire edges:

| Measurement | Seconds |
|---|---:|
| Workload load | 0.104157 |
| Current CPU exact candidate path | 13.497715 |
| Edge setup H2D | 0.023977 |
| CUDA JIT warm-up | 0.396380 |
| Point H2D | 0.081883 |
| GPU kernel | 0.060570 |
| Result D2H | 0.022149 |
| GPU transfer + kernel pipeline | 0.164670 |
| CPU float64 final candidate judge | 0.284145 |
| Steady GPU-screen + CPU-judge end to end | 0.448815 |
| Cold end to end including setup/JIT | 0.869172 |
| Measured steady core speedup | **30.074x** |

A final-code repeat on the same arrays measured CPU 21.699938 s, point H2D 0.162550 s,
kernel 0.059818 s, result D2H 0.024409 s, CPU judge 0.277496 s, and steady end to end
0.524759 s: 41.352x. The slower 30.074x run is retained above as the conservative reported
result rather than selecting the faster repeat.

Equivalence against the current CPU oracle:

- Edge position: exact
- Maximum clearance delta: 0 m
- Maximum nearest-wire XYZ delta: 0 m
- Maximum segment parameter delta: 0
- Per-tree minimum, winning point/edge, and danger count: exact for all 17 trees
- Additional CUDA adversarial smoke covered tolerance ties, equal distances, zero-length segments,
  endpoints, and coordinates near 1,000,000; all fields matched the CPU oracle exactly.

The prototype currently returns one winning edge from an all-edge CUDA screen, then recomputes
that edge with NumPy float64. Dataset equivalence is strong evidence, but it is not a mathematical
certificate that finite-precision GPU screening can never exclude another winning/tied edge.
Production integration therefore remains disabled. V1 production work must return a conservative
candidate set with a provable lower-bound certificate and route every uncertified point through the
existing CPU exact fallback before it may publish authoritative clearance.

Even if both measured per-span and final clearance time disappeared, the current 83.021 s run would
only fall by about 11.728 s, an upper-bound end-to-end speedup of about 1.16x. The requested full-run
2x target cannot be claimed from GPU clearance alone.

## Editor findings

`FULL_TREE_REBUILD_AFTER_EDIT=YES`

Every mutation currently rebuilds the in-memory final reconciliation/display projection and samples
the display identities. This does **not** rerun Ground, Wire, Tree segmentation, DBH, or clearance
science at edit commit time.

`FULL_VIEWER_RELOAD_AFTER_EDIT=YES`

This means a full editor-data payload fetch plus `setData`/WebGL buffer reconstruction, not a browser
navigation. It is the dominant 4.5–5.9 s delete/merge/undo cost.

The small lasso is correctly scoped and fast. The large lasso is dominated neither by raw IO nor by
point-in-polygon alone; full membership/snapshot traversal, array construction, and projection make
up most of the remaining 1.5 s.

## Structural findings

`IO_DUPLICATION_FOUND=YES`

- The same-source cold evidence recorded `SOURCE_FULL_PASS_COUNT=4`: three role-preparation passes
  plus one Ground workset pass.
- Immutable multi-gigabyte files are SHA-256 validated repeatedly during the same logical run.
- Viewer and export consumers rematerialize data already available in final artifacts.

`FULL_ARRAY_COPIES_FOUND=YES`

- Reconciliation repeatedly creates unique/hash/grouping arrays over millions of members.
- Final-result hydration recreates identity arrays instead of loading a directly indexed form.
- Viewer/editor payload creation converts packed numeric arrays into Python/JSON lists.
- Editor commit and refresh recreate complete display samples and identities after local mutations.
- Export recomputes revision-bound metrics from full final membership.

## Top 10 user-visible bottlenecks

`TOP_10_USER_VISIBLE_BOTTLENECKS=`

1. Cold source/role preparation: 291.893728 s inside the 421.867190 s historical cold run.
2. Cold vegetation role tile-index build: 165.913984 s.
3. Current two-span analysis: 83.020718 s end to end.
4. Cold source role preprocessing: 65.822421 s.
5. Current single-span analysis: 40.912211 s end to end.
6. Global reconciliation: 20.512594 s uninstrumented; 37.271406 s under cProfile.
7. Cold point-cloud import through usable overview: 20.744295 s effective component sum.
8. Repeated immutable-artifact hashing/acceptance: 12.553 s cumulative under cProfile.
9. Per-span plus final clearance: 11.728020 s in the current two-span wall run.
10. Cold viewer first visible frame: 6.739 s; editor point delete is close behind at 5.933420 s.

## Optimization candidates

`CPU_OPTIMIZATION_CANDIDATES=`

1. Replace reconciliation's millions of repeated `np.unique` calls with one deterministic global
   sort/index/group pass, retaining current topology order, tie rules, persistent IDs, and exact
   final membership.
2. Memoize validated fingerprints for immutable, content-addressed artifacts for the lifetime of a
   run/process; validate the manifest binding rather than rehashing identical files at each layer.
3. Stream one decoded source traversal into bounded Ground/Wire/Vegetation role consumers or a
   durable shared primitive index. Compare every authoritative role membership before acceptance.
4. Profile and vectorize remaining tree-stage hot loops only after reconciliation and hashing work.

`GPU_OPTIMIZATION_CANDIDATES=`

1. Convert the prototype from a single winner into a conservative candidate-set screen using
   float64 bounds, tolerance expansion, and explicit uncertainty detection.
2. Preserve the existing CPU exact judge and canonical tie ordering as authority; fall back for
   every uncertified point.
3. Keep `AUTO / CPU / GPU`, runtime capability detection, and memory-derived chunk sizing.
4. After clearance acceptance, profile only measured high-volume neighbor/radius queries and
   reductions; do not move frozen science wholesale to CUDA.

`VIEWER_OPTIMIZATION_CANDIDATES=`

1. Serve packed binary/typed numeric buffers rather than multi-megabyte nested JSON lists.
2. Deliver a progressive first useful frame while background buffers finish loading.
3. Add revisioned tree/point deltas so edits update affected WebGL buffers instead of clearing and
   recreating the entire scene.
4. Retain selection and camera state across delta commits.

`EDITOR_OPTIMIZATION_CANDIDATES=`

1. Maintain original-index-to-tree and tree-to-display-row indexes across edits.
2. Apply deterministic local deltas for delete/merge/undo; rebuild only affected records and display
   identities.
3. Screen large lasso candidates spatially before projecting points; reuse projected coordinates.
4. Cache final metrics by editor revision and invalidate only affected trees before export.

## Recommended optimization order

`RECOMMENDED_OPTIMIZATION_ORDER=`

1. **Source pass and cache validation V1.** Add exclusive timers first, then remove repeated full
   source traversals and same-run rehashes. This is the dominant cold-run opportunity.
2. **Reconciliation V1.** Replace repeated `np.unique`/hash operations with one deterministic indexed
   algorithm. Require byte/exact membership, persistent-ID, ordering, and clearance equivalence.
3. **Editor/viewer delta V1.** Remove full display reconstruction from delete/merge/undo and use a
   revisioned delta protocol. Require exact journal replay and restart equivalence.
4. **GPU clearance production V1.** Add conservative certification plus CPU exact fallback, then wire
   the accepted backend behind `AUTO / CPU / GPU`. Re-run transfer, kernel, judge, and full-pipeline
   benchmarks on the same source.
5. **Tree-stage CPU V1.** Profile its internal kernels and optimize only the measured hot path while
   preserving authoritative CROWN_STEM_HYBRID membership.
6. **CPU/GPU pipeline overlap (Gate 3).** CPU prepares the next span, GPU screens the current span,
   and CPU exact-judges/persists the prior span with bounded queues and cancellation.
7. **Multi-span scheduler (Gate 4).** Add CPU producers and one device queue; reconcile in fixed
   topology order. Release `PARALLEL` only after SERIAL/PARALLEL equality for every authoritative
   array, record, persistent identity, clearance, and report field.

## Verification

- Ruff: PASS for the standalone benchmark and focused test.
- Python compile: PASS.
- Final-code real CUDA repeat: PASS, 17/17 tree summaries exact and all compared arrays exact.
- Focused/regression pytest: 37 passed in 4.75 s. Coverage included risk-threshold dynamic
  reclassification, region lasso performance, clearance publication order, candidate backend,
  fast baseline clearance, full-resolution clearance, and manual-span clearance equivalence.
- The first regression invocation had six fixture setup errors because the machine's shared pytest
  temporary directory was inaccessible. The exact same selection passed when rerun with a fresh,
  repository-scoped temporary root and cache provider disabled. This was an environment error, not
  a test assertion failure; the temporary root was removed afterward.
- No release package or EXE was rebuilt.
