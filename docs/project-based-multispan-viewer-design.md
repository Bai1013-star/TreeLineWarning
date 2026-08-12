# Project-Based Multi-Span Viewer Design

**Status:** design review candidate  
**Scope:** architecture and planning only; no production implementation is authorized by this document.

## 1. Problem Statement

TreeLineWarning currently exposes a validated scientific workflow through a manual, single-span entry point. The operator opens one LAS file, makes two approximate pole selections in a sampled Plotly browser view, and runs a complete analysis for that one pair. The workflow is scientifically useful, but its entry model is not yet suitable for a transmission-line project:

- a top-down sample can obscure a thin, vertically oriented pole;
- the primary selector treats Pole A and Pole B as disposable inputs instead of persistent physical assets;
- adjacent spans require the shared pole to be selected again;
- no persistent line topology or per-span status exists; and
- the UI is a localhost browser page rather than the future Windows desktop product.

The target is a project-based desktop application in which a source point cloud is authoritative, a full-scene viewer is an interactive multi-resolution representation, towers are persisted once, adjacent spans reference the same tower records, and every span can independently reuse the existing scientific workflow.

This is an entry-workflow and orchestration redesign. It must not replace, recalibrate, or otherwise alter the validated scientific stages.

## 2. Current Architecture Audit

### Selector and local refinement

The audited primary selector is `algorithms/manual_span/server.py`, launched by `scripts/run_manual_single_span.py` and `RUN_MANUAL_SINGLE_SPAN.bat`.

- It is a `ThreadingHTTPServer` bound to `127.0.0.1` on an ephemeral port and opens the system browser. There is no desktop shell.
- The HTML/CSS/JavaScript is embedded in `_selector_html()` in the Python server. Plotly JavaScript is served locally at `/plotly.js`.
- The server exposes a compact local API: `/points`, `/state`, `/local_points`, `/mode`, `/click`, `/refine`, `/reselect`, `/confirm`, `/analyze`, and `/threshold`.
- The primary view is a top-down XY Plotly scatter. The browser receives at most 80,000 display points; the secondary Plotly 3D view receives at most 30,000. The source’s real coordinates are preserved in the payload, but the browser does not receive a source-row identifier for the click.
- A click is validated against source bounds in the source-LAS absolute frame. It supplies XY and uses the supplied Z or the nearest displayed point’s Z as an approximate seed. It is not treated as the mathematical pole center.
- `ManualSingleSpanRun.local_raw_for_xy()` streams the original LAS in one-million-point chunks, extracts a full-resolution local XY box across the source Z extent, and caches `xyz_absolute`, RGB, and `original_index` in `00_input/local_raw/`.
- `refine_pole_from_raw_cloud()` in `algorithms/manual_span/span.py` evaluates the local raw cloud for a continuous narrow vertical shaft. It records click and refined geometry, support, verticality, linearity, footprint, diagnostics, and a defensible/failure result. Failures are explicit (for example sparse points, insufficient height, wide footprint, broken continuity, or non-shaft-like geometry); the code does not silently snap to an arbitrary nearby point.
- The existing test suite covers real-coordinate serving, source-SHA binding, selector state transitions, reset behavior, raw-local endpoint behavior, defensible refinement, and refinement failures in `tests/vegetation/test_manual_selector.py` and `tests/vegetation/test_manual_span.py`.

The repository also contains `scripts/crop_selector.py`, a separate Streamlit/Plotly crop tool. It is not the manual single-span entry application. No pywebview, PySide6, PyQt, or Qt WebEngine dependency is declared in `pyproject.toml`.

### Existing span representation

`algorithms/poles/topology.py` already defines an algorithm-facing `SpanRecord`: it carries an integer `span_id`, a name, algorithmic start/end pole candidate IDs, absolute endpoints, pole-top elevations, length, direction, corridor width, halo, local core/halo bounds, and quality. The generic pole-topology code can build adjacent spans from a graph of automatically validated pole candidates.

The manual path wraps that contract rather than creating a persistent user project model:

- `ManualRefinedPole` is labelled only `A` or `B` and contains click/base/top geometry and refinement evidence.
- `build_manual_span_record()` creates one `SpanRecord` with `line_id=0`, `span_id=0`, `ManualLine-Span001`, and synthetic candidate IDs `-1`/`-2`.
- `ManualSingleSpanRun.refine_and_build_span()` accepts exactly two clicks, refines them, writes the selection payload, and then stores the resulting `SpanRecord` on the live run.
- Downstream wire and tree folders are hard-coded as `span_001` inside the single run. This is an internal legacy convention, not a project-level stable span identifier.

Therefore `SpanRecord` is valuable as the algorithm adapter input, but it is not a suitable replacement for a Tower Registry or project-owned `S001`, `S002`, … identity.

### Current run and artifact ownership

The launcher creates timestamped runs below `artifacts/manual_single_span_runs/<source-stem>_<timestamp>/`. A run owns stages `00_input` through `11_report`:

```text
00_input/        inspection, display sample, local RAW caches
01_preprocess/   weighted representatives and mapping
02_ground/       ground/HAG artifacts
03_selection/    selector journal / confirmed selection and checkpoint
04_regions/      halo, wire, and vegetation regions
05_wire/span_001/
06_tree/span_001/
07_clearance/
08_metrics/
09_risk/
10_visuals/
11_report/
```

Checkpoints are keyed by source SHA-256, source ID, configuration digest, upstream digest, and stage in `algorithms/spans/checkpoint.py`. They are intentionally fail-closed: a mismatch invalidates reuse.

The audited completed example contains a 105,055,350-point LAS and a complete per-run report chain, including wire, tree, clearance, risk, Plotly visualizations, Excel, and HTML report output. It also shows that tree and wire records currently carry the legacy integer `span_id=0` and `span_name=ManualLine-Span001`.

There is one storage-version discrepancy to preserve during migration planning: current code intends to write `03_selection/manual_span_selection.json` on confirmation, but the audited older completed example contains only `manual_selector_state.json` in that directory while its `run_summary.json` contains the refined A/B and span data. A pending selector journal must never be treated as a confirmed project topology. Any future legacy import must validate an explicit confirmed artifact or require operator confirmation from the run summary.

### Source and coordinate identity

The manual run computes a full-file SHA-256, derives `manual_span_<sha-prefix>` as `source_id`, records LAS header bounds/scales/offsets, and uses `bounds_min_xyz` as a local origin. The manual selection is persisted in `SOURCE_LAS_ABSOLUTE`; algorithmic corridor bounds are local to the selected origin. `original_index` is an `int64` source-row identity preserved through raw-local recovery and full-resolution analysis.

The repository also has a stronger reusable identity model in `core/contracts/frv.py`:

- `SourceIdentity` separates a stable content identity (`source_content_identity_digest`) from an auditable access/path record (`source_identity_record_digest`);
- it binds file SHA-256, canonical content identity when available, point count, bounds, coordinate frame, units, origin context, and relocation/equivalence audit records; and
- `algorithms/io/source_provider.py` verifies source identity during streamed loading and preserves `original_index` as the source-domain identity.

The project model should adopt this stronger contract rather than spread another path-only identity scheme.

### Existing visualization and UI stack

The manual selector and most review artifacts use Plotly. `scripts/build_downstream_flow.py` builds static HTML review pages around the same single-span artifacts. The repository has no existing WebGL LOD point-cloud engine, packaged desktop shell, or all-project tree UI. Current declared runtime dependencies are Pillow, laspy, NumPy, and Plotly.

## 3. Design Goals

1. Make a **Project and Transmission-Line Topology Initialization** subsystem the first-class entry workflow.
2. Keep the original LAS/LAZ authoritative and display it interactively without loading every point into RAM or GPU memory at once.
3. Make an operator click only near a physical tower; retain and reuse existing high-resolution local pole refinement.
4. Persist each confirmed physical tower once in a Tower Registry, with stable IDs such as `T001`.
5. Persist each project span once with stable IDs such as `S001`, and reference tower IDs instead of duplicating Pole A/Pole B domain records.
6. Make each span independently runnable, resumable, reviewable, exportable, and status-bearing while retaining project-level navigation.
7. Wrap the current single-span pipeline behind an adapter before considering scientific refactoring.
8. Keep the existing Plotly selector callable as a legacy/debug/fallback path until the new viewer is accepted.
9. Keep prediction optional and unavailable by default; static analysis must work without prediction software, model files, or a second computer.
10. Leave a credible Windows `.exe` path without installing, converting, packaging, or changing dependencies in this design stage.

## 4. Non-Goals

This design does not:

- implement Potree, PotreeConverter, pywebview, PySide6, Qt, a desktop executable, or a project UI;
- convert the current 105M-point source or generate a viewer cache;
- alter ground/HAG, wire, vegetation, tree, clearance, risk, thresholds, or reports;
- re-run a point-cloud stage or change existing outputs;
- delete or modify the manual selector;
- define a monolithic all-spans scientific algorithm;
- implement, train, estimate, or fabricate vegetation-growth/future-risk predictions;
- transfer prediction code from another computer; or
- change Git state other than this design document.

## 5. Proposed System Architecture

The V1 architecture is deliberately layered. The viewer never becomes the source of scientific truth, and the project registry never becomes a replacement point-cloud processor.

```text
Authoritative LAS / LAZ (read-only)
          |
          +--> Project source identity and source reference
          |
          +--> Viewer-cache builder (future, reproducible, non-authoritative)
          |       +--> Potree static hierarchy --> Potree viewer
          |
          +--> Existing local RAW extractor --> existing pole refinement
          |
Project service / local API
  +--> project.json + tower registry + span registry + run index
  +--> topology controller (sequential Add Next Tower)
  +--> span-to-existing-pipeline adapter
  +--> optional prediction exchange boundary
          |
Desktop shell --> project tree + action panels + viewer viewport
```

The project service owns all mutation and validation. The JavaScript viewer is a client: it requests a display cache, reports an approximate pick, displays markers/status, and never writes `project.json` directly. The scientific pipeline remains the only owner of its established artifacts and result contracts.

## 6. Viewer Technology Decision

### Recommendation: Potree / PotreeConverter for V1 viewer rendering

Use a pinned, reviewed Potree distribution as the future primary selection surface, with PotreeConverter-generated project viewer caches. Potree is an open-source WebGL viewer explicitly designed for large point clouds and its project documentation describes a multi-resolution octree renderer, a Windows converter, local/static hosting, point-cloud loading, point budget controls, RGB/elevation material options, measurements, camera controls, and annotation examples. Its examples include point clouds far larger than the present 105M-point source. See the [Potree project documentation](https://github.com/potree/potree) and its [annotations example](https://github.com/potree/potree/blob/develop/examples/annotations.html).

This is a viewer decision, not an analysis-format decision:

- The LAS/LAZ remains the authoritative original source for refinement and science.
- A Potree cache is a derived, reproducible presentation index. It is never a substitute for source `original_index`, source SHA, or raw point recovery.
- **Full dataset availability** means the project retains and can validate access to the entire source dataset.
- **Current display LOD** means only a view-dependent subset is streamed and rendered under a point budget. A picked rendered point is a good approximate seed, not proof of an exact original source row.

Potree must be evaluated in a small proof-of-concept before full conversion, including WebView2 GPU compatibility, pick reliability, color modes, marker overlay ergonomics, and offline static serving. Pin the exact viewer and converter version, cache-generation settings, checksums, and license review in the future implementation manifest rather than tracking an unpinned upstream branch.

### Required viewer interaction contract

The custom project layer must provide:

- rotate, zoom, pan, top view, side view, reset, and later a line-direction view;
- RGB, elevation, and HAG only when the relevant attribute/result is actually available;
- custom Tower markers and span corridor overlays from the registry;
- a pick event reporting viewer-local XYZ, displayed LOD/node metadata, viewport/camera context, and cache identity;
- a bridge that converts viewer-local XYZ to source-absolute XYZ before the backend validates it; and
- focus actions for towers/spans without modifying selection or science data.

Do not require Potree to preserve original LAS point order or use a display-pick `original_index`. The backend must retrieve local high-resolution source points by a bounded source-coordinate query and run the existing refinement routine.

### COPC position

COPC is a future alternative or complementary source/cache option, not a first implementation requirement. COPC is a spatially organized LAZ format with hierarchy metadata for resolution/spatial selection; its specification is available at [copc.io](https://copc.io/). It may reduce duplication and improve future spatial access, but it changes source/cache policy and reader compatibility questions. V1 should use a Potree-derived viewer cache only after the POC succeeds and should not require source conversion to COPC.

## 7. Project Model

The project is a persistent, versioned aggregate. `project.json` is the small authoritative manifest; large or independently mutable collections have their own records. The format starts at `treeline-project-v1` and every manifest is written atomically (temporary sibling, validation, rename).

Minimum project manifest fields:

```json
{
  "schema_version": "treeline-project-v1",
  "project_id": "uuid",
  "project_name": "Line A",
  "created_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC",
  "source_ref": "source/source-reference.json",
  "viewer": {"active_cache_id": null, "state": "NOT_BUILT"},
  "tower_registry_ref": "towers/towers.json",
  "span_registry_ref": "spans/spans.json",
  "run_index_ref": "runs/run-index.json",
  "prediction_capability": {"status": "NOT_CONNECTED"},
  "next_tower_sequence": 1,
  "next_span_sequence": 1
}
```

`project_id` is immutable. Display IDs are deterministic project-local IDs (`T001`, `S001`) allocated monotonically and never reused after deletion; deleted records become archived/tombstoned entries rather than letting a different physical object inherit an old identity. A project stores references to source, cache, registry, and reports, not copies of all point data.

## 8. Tower Model

Each confirmed physical tower has exactly one Tower Registry record. The domain record is independent of every span.

```json
{
  "schema_version": "treeline-tower-v1",
  "tower_id": "T002",
  "display_name": "T002",
  "state": "CONFIRMED",
  "source_content_identity_digest": "sha256...",
  "coordinate_context": {
    "frame": "SOURCE_LAS_ABSOLUTE",
    "units": "m",
    "origin_xyz": [0.0, 0.0, 0.0]
  },
  "approximate_click_xyz_absolute_m": [0.0, 0.0, 0.0],
  "refined_geometry": {
    "base_xyz_absolute_m": [0.0, 0.0, 0.0],
    "top_xyz_absolute_m": [0.0, 0.0, 0.0],
    "height_m": 0.0
  },
  "refinement": {
    "status": "DEFENSIBLE",
    "evidence": {},
    "quality": "MANUAL_CONFIRMED"
  },
  "notes": null,
  "revision": 1,
  "created_at": "ISO-8601 UTC",
  "updated_at": "ISO-8601 UTC"
}
```

The refinement evidence stores the existing diagnostics plus future algorithm/version identifiers, local-recovery bounds or artifact reference, and confirmation time. It must not contain an unbounded raw point dump. A correction creates a new revision and marks affected spans/results stale; it does not silently alter the provenance of an already completed analysis run.

Failed or rejected selection attempts are stored in a compact selection audit log with their approximate click, reason, diagnostics, and timestamp. They are not confirmed Tower Registry records and cannot form a span. This preserves the operator’s approximate click and failure reason while preventing an accidental tower from becoming topology.

## 9. Span Model

A project span is topology, status, and ownership metadata. Its endpoint fields are references, never embedded independent Pole A/Pole B records.

```json
{
  "schema_version": "treeline-span-v1",
  "span_id": "S002",
  "display_name": "S002  T002 -> T003",
  "from_tower_id": "T002",
  "to_tower_id": "T003",
  "topology_state": "CONFIRMED",
  "geometry": {
    "definition_revision": 1,
    "length_m": 0.0,
    "direction_xy": [0.0, 0.0],
    "core_bounds_local_xyz": {"min": [0, 0, 0], "max": [0, 0, 0]},
    "halo_bounds_local_xyz": {"min": [0, 0, 0], "max": [0, 0, 0]},
    "geometry_digest": "sha256..."
  },
  "analysis": {
    "state": "NOT_STARTED",
    "active_run_id": null,
    "stages": {
      "ground": "NOT_STARTED",
      "wire": "NOT_STARTED",
      "vegetation": "NOT_STARTED",
      "trees": "NOT_STARTED",
      "clearance": "NOT_STARTED",
      "risk": "NOT_STARTED",
      "report": "NOT_STARTED"
    }
  },
  "artifact_index_ref": "spans/S002/artifact-index.json",
  "prediction": {"status": "NOT_CONNECTED"}
}
```

The `geometry` values are derived from the referenced, confirmed tower revisions and the frozen corridor configuration. They are stored for review, cache invalidation, and efficient navigation, with a digest binding the source/tower revisions/configuration. A span is marked `STALE` if an endpoint tower revision, source identity, or corridor configuration changes. It can retain its historical runs but may not claim that their result is current.

The adapter may generate legacy Pole A and Pole B records inside a specific run because the existing pipeline requires them. Those generated compatibility artifacts are not a second tower registry and are always mapped back to `from_tower_id`/`to_tower_id` and their revisions.

## 10. Source / Coordinate Identity

The project must create `source/source-reference.json` before any cache, tower, or span can be confirmed.

- Prefer the existing `SourceIdentity` content/record digest model where the project source path can be brought under that contract. At a minimum retain file SHA-256, byte size, LAS header metadata, point count, bounds, coordinate frame, units, origin, source content identity, and source identity record digest.
- Keep `source_content_identity_digest` as the stable binding across path relocation and use `source_identity_record_digest` for the audited current access record. Do not use a Windows path as source identity.
- Keep source-absolute XYZ as the canonical coordinate representation for user clicks, tower geometry, raw recovery, and legacy adapter construction. The project declares an explicit source frame string and units; it must not invent CRS conversions.
- Store the source origin used by existing local science. Viewer rendering uses `viewer_local_xyz = source_absolute_xyz - origin_xyz`; every pick sent to the backend contains the viewer cache identity and is transformed back to source-absolute XYZ before validation.
- Preserve `original_index` as a source-row identity in raw/scientific artifacts. It is not an index into Potree nodes and must not be reconstructed from visual draw order.

If a referenced file moves, the app asks the user to locate it, verifies identity, and records an approved relocation/equivalence audit. A digest mismatch blocks tower refinement and analysis. It must not quietly analyze a same-named file.

## 11. Viewer Cache Strategy

The viewer cache is derived data and has an explicit lifecycle:

```text
source/source-reference.json             # authoritative identity and access reference
viewer/caches/<cache-id>/cache-manifest.json
viewer/caches/<cache-id>/potree/          # derived hierarchy and static assets
viewer/active-cache.json                  # small pointer after validation
```

`cache-id` is a digest of the source content identity, viewer format, pinned converter version, conversion settings, coordinate transform declaration, and cache schema version. `cache-manifest.json` binds those values, declares bounds/origin/attribute availability, stores output file digests, and is written last as the cache-complete marker.

Rules:

- A cache mismatch, absence, corruption, or partial conversion makes the viewer unavailable/rebuildable; it never invalidates the original source or completed scientific runs by itself.
- Cache generation writes to a temporary sibling directory and atomically publishes only after validation. Cancellation leaves no active pointer to a partial cache.
- The viewer streams and renders a point budget. The point budget, LOD, density, and current GPU residency are presentation settings, not scientific settings.
- RGB is available only when source/cache dimensions support it. Elevation is always a presentation derivation from coordinates. HAG is displayed only from a specifically selected, identity-bound derived artifact; it is not fabricated from the viewer cache.
- The first POC uses a small, disposable sample/cache. Full source conversion is a later operator-visible job with disk-space estimate, progress, cancel, cache manifest, and no scientific-stage execution.

## 12. Tower Selection UX

The primary viewport is the full-scene viewer. A left project tree and small action panel stay visible; they do not replace the viewport.

Minimum camera actions are Free 3D View, Top View, Side View, and Reset View. Once two confirmed towers establish a direction, Line-Direction View may orient the camera along the active segment. The color menu exposes only available modes: RGB; elevation; and HAG if an identity-bound HAG visualization layer exists.

The first action is **Add First Tower**. The operator clicks near a pole, not its exact center. The viewer sends an approximate local point with cache/LOD/camera metadata. The backend transforms and validates the seed, retrieves full-resolution local raw points, runs refinement, and returns a review card over the local 3D crop:

- approximate click and refined base/top marker;
- height, footprint, support, shaft diagnostics, and quality;
- Confirm, Reject, and Reselect actions; and
- an unambiguous failure reason if no defensible tower was found.

Only Confirm commits `T001`; then the marker and registry/tree node are persistent. Reject preserves an audit attempt but creates no confirmed tower. Reselect keeps the project/tower registry unchanged and returns to click mode.

## 13. Tower Refinement Interface

Define a small project-layer interface around the existing behavior rather than a new pole detector.

```text
TowerRefinementRequest
  project_id, source content identity, viewer cache id/version
  click_xyz_viewer_local_m, click_xyz_source_absolute_m
  selection mode, requested search radius, client/camera metadata

TowerRefinementResult
  status: DEFENSIBLE | FAILED
  approximate click; refined base/top geometry when defensible
  quality/evidence/diagnostics; bounded local-region reference
  reason code when failed; algorithm and input-contract versions
```

The initial adapter calls `ManualSingleSpanRun.local_raw_for_xy()` semantics and `refine_pole_from_raw_cloud()` semantics. It must return the existing diagnostic evidence in a project-neutral envelope. The future project service may extract these functions from the legacy run object, but the acceptance condition is behavioral reuse, not a rewrite.

Failure behavior is mandatory:

- invalid cache/source binding, out-of-bounds pick, missing source, malformed coordinate context, or local recovery error returns `FAILED` and does not create a tower;
- refinement failures preserve the approximate seed and specific reason in the attempt audit;
- no result may substitute the nearest displayed point, a cache node centroid, or random raw point as a confirmed tower; and
- a changed source identity or tower revision invalidates unfinished refinement proposals.

## 14. Multi-Tower Workflow

The normal linear workflow intentionally encodes operator intent so a shared tower is never selected twice:

1. Create/open a project and validate the source reference.
2. Open the full-scene viewer and select **Add First Tower**. Confirm `T001`.
3. Select **Add Next Tower**. The active predecessor is `T001`; confirm the new `T002`.
4. The sequential controller immediately creates `S001` with `from_tower_id=T001` and `to_tower_id=T002`, displays it, and offers Undo Span/Tower before analysis.
5. Select **Add Next Tower** again. The active predecessor is now `T002`; confirm `T003`; the controller creates `S002 = T002 -> T003`.
6. Continue through `T003 -> T004`, etc., always returning to the same project and scene.

The controller records its active line/continuation tower separately from tower identity. A free-selection mode may later create branches, reversed spans, disconnected tower groups, or manually selected endpoint pairs; it is not required for the sequential V1 acceptance path. Editing a confirmed intermediate tower must show all affected spans before commit.

## 15. Multi-Span Workflow

Each project span is a separate analysis unit:

```text
Project Line A
  T001 --- S001 --- T002 --- S002 --- T003
               \               /
       independent runs, artifacts, statuses, reports
```

Selecting `S001` activates only `S001`'s status, corridor overlay, review pages, and runs. Selecting `S002` does the same independently. No all-spans processing pass is introduced.

At project level, a later aggregate reads immutable/current span summaries and may display counts such as total towers, total spans, analyzed spans, risk distribution, most critical current span, and minimum clearance. Aggregation is read-only; it neither changes a span result nor forces a span to run.

## 16. Span-to-Existing-Pipeline Adapter

Introduce a thin future `ProjectSpanRunAdapter` at the boundary between project topology and the existing manual pipeline. It is an orchestration adapter, not a scientific rewrite.

For one project span it will:

1. Load the project’s validated source reference, `from_tower_id`, `to_tower_id`, their immutable revisions, and corridor configuration.
2. Derive the algorithm-facing `SpanRecord` via the existing `build_manual_span_record()` geometry rules, while retaining the project’s stable `Sxxx` identity in adapter metadata.
3. Materialize compatibility `ManualRefinedPole` A/B views from the referenced Tower records only for the run input and create the current `manual_span_selection`-compatible provenance payload.
4. Run the existing initialization, regions, wire, tree, clearance, risk, visuals, and report stages in an isolated run root.
5. Write a project-owned run manifest mapping `project_id`, `span_id`, tower revision refs, project geometry digest, source identity, configuration digest, legacy internal `span_001`, and result/artifact paths.
6. Promote a completed run to the span’s `active_run_id` only after integrity/status validation.

For the first adapter release, each project span may retain legacy-internal `span_001` beneath its own run root because existing wire/tree code expects that layout. `S001` and `S002` remain distinct by their project folders and run manifests. The user will never see those internal integer IDs as project identity.

Source-level preprocessing/ground reuse is a performance optimization, not a prerequisite for the data model. It can be introduced only after an explicit provenance and concurrency design proves that it reuses validated, identity-matched artifacts without changing results. The safest first adapter is isolated and compatible; it preserves independent resume behavior and the current science contracts.

## 17. Project Storage Layout

Recommended default layout:

```text
<ProjectName>/
  project.json
  source/
    source-reference.json
    managed/                         # empty unless user explicitly chooses managed copy
  viewer/
    active-cache.json
    caches/<cache-id>/
      cache-manifest.json
      potree/
  towers/
    towers.json
    selection-attempts.jsonl
  spans/
    spans.json
    S001/
      span.json
      artifact-index.json
      runs/<run-id>/
        project-span-run.json
        legacy/                       # current stage layout, including span_001 where needed
      prediction/                     # reserved; normally absent/empty in V1
    S002/
      ...
  reports/
    project-report-index.json         # reserved for aggregation
  prediction/
    project_prediction_manifest.json  # reserved; NOT_CONNECTED by default
  runs/
    run-index.json
  logs/
```

### Source ownership decision

**Default: reference the original LAS/LAZ read-only.** Store its resolved path, file size, source identity, and header/coordinate metadata in `source-reference.json`; do not copy a multi-gigabyte point cloud just because a project is created.

Two explicit alternatives remain available later:

- **Project-managed source:** after space estimate and operator confirmation, copy/import the source once to `source/managed/`, hash it, and make that managed file the validated access path. This supports portable archival projects.
- **External-reference relocation:** when the original moves, locate and validate a content-equivalent source, then update only the audited access record.

Never silently copy original LAS/LAZ files into every project, every span, or a viewer cache. Cache and run paths must be project-relative, while source identity remains independent of the machine path.

## 18. Result / Artifact Ownership

Ownership is intentionally separated:

| Owner | Owns | Must not own |
| --- | --- | --- |
| Source reference | source identity and access audit | derived cache or mutable analysis result |
| Viewer cache | derived LOD/static render data | authoritative science points/results |
| Tower Registry | confirmed physical tower/revision/evidence | per-span copied tower records |
| Span Registry | topology refs, derived corridor, status, active-run pointer | full scientific artifacts |
| Span run | legacy-compatible stage artifacts, result summary, report refs | another span’s output |
| Project summary | aggregation/indexes | recomputation or mutation of spans |
| Prediction area | optional exchange manifests/results | required static workflow state |

Existing `artifacts/manual_single_span_runs` remain untouched. A project may register a legacy run as an external, read-only historical reference after explicit source and confirmation validation, but should not move it or claim an unconfirmed pending journal is a valid topology record.

## 19. Project Tree UI

The desktop shell hosts a left navigation tree and an always-visible main viewer viewport:

```text
Project: Line A
  Source Point Cloud
  Towers
    T001
    T002
    T003
  Spans
    S001  T001 -> T002
      Ground | Wire | Vegetation | Trees | Clearance | Risk | Prediction | Report
    S002  T002 -> T003
      Ground | Wire | Vegetation | Trees | Clearance | Risk | Prediction | Report
  Reports
```

Selecting a tower focuses the viewer on its refined geometry and evidence marker. Selecting a span focuses its corridor/core/halo overlay. Selecting an existing result stage opens a result/review page or overlay associated with the active run; it must clearly label stale, incomplete, failed, and historical results. The Prediction node is always visible and honestly displays `NOT_CONNECTED` when no provider is installed.

## 20. Desktop/EXE Architecture

### Option A: Python + Potree + localhost API + pywebview + PyInstaller

This option retains the existing Python-local-server and HTML/JavaScript skills. The app starts a loopback-only project API/static server, opens one pywebview window, and loads the project UI and Potree viewer from that server. pywebview offers a native webview wrapper, a local HTTP server/JavaScript bridge, and Windows support; its own documentation notes PyInstaller packaging and recommends excluding unused GUI backends to control package size. See [pywebview introduction](https://pywebview.idepy.com/en/guide/) and [freezing guidance](https://pywebview.idepy.com/en/guide/freezing). PyInstaller bundles Python and application dependencies for same-platform distribution; Windows packages must be built on Windows. See the [PyInstaller manual](https://pyinstaller.org/en/stable/index.html).

Advantages: lowest migration cost; direct reuse of the current local HTTP architecture; normal browser development/debugging; static Potree assets work naturally; no second Python-to-Qt bridge is required. Risks: Windows WebView2/runtime/GPU compatibility needs an installer preflight; localhost API security and process shutdown must be designed; package assets and native scientific dependencies still need test coverage.

### Option B: Python + Potree + PySide6/Qt WebEngine

This replaces the shell with a Qt application and embedded Chromium-based Qt WebEngine. It can use `QWebChannel` or a restricted custom URL scheme for the Python/JavaScript bridge. Qt documents custom URL schemes and their explicit security flags, and Windows deployment requires WebEngine libraries, plugins, resources, and locales; deployment commonly uses `windeployqt`. See [Qt WebEngine deployment](https://doc.qt.io/QT-6/qtwebengine-deploying.html) and [custom URL scheme handling](https://doc.qt.io/QT-6/qwebengineurlschemehandler.html).

Advantages: a controlled bundled engine, richer native desktop widgets, and stronger ability to avoid a local HTTP transport later. Costs: a larger installer/runtime, more involved Qt deployment/debugging, a more complex bridge, and a greater rewrite from the current HTTP/HTML entry point.

### V1 recommendation

Choose **Option A: Python backend + Potree frontend + loopback local API + pywebview shell + PyInstaller**. It fits the present server architecture and permits focused validation of the large-cloud viewer before committing to a large Qt distribution. The future installer must detect or provision a supported WebView2 runtime, make the renderer choice explicit, package only the selected backend, bind only to loopback, and provide a diagnostic/log export path. Qt remains the fallback if field tests prove WebView2/WebGL behavior or native-shell requirements unacceptable.

## 21. Failure and Recovery Behavior

| Condition | Required behavior |
| --- | --- |
| Referenced source missing | Mark source unavailable; disable refinement/new analysis; prompt locate-and-verify; preserve project data. |
| Source identity mismatch | Block source use and cache/run reuse; never substitute a same-name file. |
| Viewer cache missing/stale | Viewer reports cache unavailable/rebuild required; static project topology and old reports remain readable. |
| Cache generation interrupted | Leave no active-cache pointer; clean only the job-owned temporary directory. |
| Viewer pick outside valid source/cache context | Reject with explicit reason; no tower draft is confirmed. |
| Pole refinement fails | Preserve the approximate click/audit, show diagnostics, allow reselect/retry/reject; create no span. |
| Tower edit after results | Increment tower revision, mark dependent spans stale, retain historical runs. |
| Span execution fails | Mark only that run/span stage failed; retain upstream valid artifacts/checkpoints; other spans remain usable. |
| App closes during run | Mark run interrupted on next open; validate checkpoints before resume; do not promote partial results. |
| Prediction absent | Display `NOT_CONNECTED`; all static functionality remains available. |
| Prediction bundle invalid | Reject it atomically with a reason; do not change static results or project topology. |

## 22. Performance Considerations

- A 100M-point source must be treated as out-of-core. Full availability is not an instruction to load full points into CPU memory, JavaScript arrays, or GPU buffers.
- The viewer's conversion cost and storage are explicit background work. Runtime navigation relies on Potree hierarchy/LOD and a configurable point budget, not on serializing the entire source through JSON.
- The project service should return compact metadata/deltas for tree nodes and markers. It must not repeatedly send full point clouds or complete run summaries to the renderer.
- Pick-to-refinement uses bounded local raw recovery. It may still stream a source with the present implementation, so local cache indexing/reuse and cancellation telemetry are later performance work; correctness must precede a new spatial index.
- Preprocessing/ground reuse belongs at a source-identity boundary and must be measured separately from per-span wire/tree/clearance work. It should not compromise independent span resumability.
- Persisted viewer camera/color/point-budget preferences are display-only and are never included in scientific configuration digests.
- POC and full-cache acceptance should record conversion time, cache bytes, first-view latency, interactive frame behavior, pick latency, refinement latency, RAM, GPU memory, and cancellation/reopen behavior on representative Windows hardware.

## 23. Backward Compatibility

- `scripts/run_manual_single_span.py`, `algorithms/manual_span/server.py`, `ManualSingleSpanRun`, existing result formats, and existing artifact roots remain operational and untouched.
- The old Plotly selector is labelled **Legacy / Debug / Fallback Selector** once the project UI is introduced; it is not deleted until project-viewer acceptance tests prove a replacement.
- The project adapter consumes current `SpanRecord` and manual refinement semantics rather than changing downstream function contracts.
- Existing single-span artifacts remain read-only. Legacy import is optional, explicit, provenance-checked, and may create a historical project reference; it never rewrites old output.
- Old projects with no prediction section load as `prediction.status = NOT_CONNECTED` by default.
- Unknown future prediction payload fields are preserved in their namespaced envelopes or ignored safely; they must not make core project/tower/span data unreadable.

## 24. Migration from Current Manual Selector

1. Leave the current launcher and Plotly server unchanged.
2. Add project/tower/span contracts and persistence without changing a scientific stage.
3. Implement the new viewer separately behind a feature flag and validate it on a small cache.
4. Add approximate-pick-to-existing-refinement bridging and compare its refinement payloads to the legacy path.
5. Add the sequential Tower Registry workflow, then create project spans.
6. Add the compatibility adapter that writes new, project-scoped isolated run roots but invokes current single-span behavior.
7. Add project result navigation and only later project summary aggregation.
8. Retain the legacy selector as an explicit fallback until the Owner accepts the new path.

No migration step may reinterpret `manual_selector_state.json` with `PENDING_UNCONFIRMED` status as a confirmed tower/span. Confirmed legacy selection payloads or an operator-confirmed reconstruction from summary data are required.

## 25. Security / File-System Safety

- The desktop API listens only on `127.0.0.1`, uses an unpredictable per-launch capability token supplied only to the owned shell, checks `Origin`/request type, and has no public network listener by default.
- Project IDs, tower IDs, span IDs, cache IDs, and run IDs are validated against strict formats. Client requests never supply arbitrary filesystem paths for server reads/writes.
- The project service resolves all project-relative paths and verifies they remain below the project root. Source paths are separately validated, read-only references.
- Source imports and managed copies require explicit user choice, free-space preflight, hashing, atomic publication, and cancellation cleanup limited to the job-owned temporary directory.
- Viewer cache assets are served only from the active cache root after manifest validation. No arbitrary static-file endpoint is exposed.
- Prediction import/export validates canonical manifest version, relative paths, file hashes/sizes, IDs, coordinate context, and source/run bindings. Archive extraction, if added, rejects absolute paths, traversal, duplicate conflicting paths, and unexpected large members before writing.
- Logs and UI must avoid exposing access tokens. Project metadata may contain source paths, so diagnostic export should require user action and disclose included paths.

## 26. Testing Strategy

Testing is layered and no expensive production point-cloud run is required for every test.

1. **Contract tests:** schema validation, stable ID allocation/no reuse, JSON round trips, revision handling, tower/span endpoint references, source identity binding, stale-state transitions, and forward-compatible prediction envelopes.
2. **Topology tests:** sequential `T001 -> T002 -> T003` creates `S001` and `S002` with `T002` referenced once; undo/reselect does not leave duplicate/shared towers; edits mark only dependent spans stale.
3. **Refinement bridge tests:** viewer-local to source-absolute conversion, request bounds/cache identity validation, success/failed refinement envelopes, and equivalence with existing raw refinement fixtures.
4. **Adapter tests:** a fixture span produces a legacy-compatible selection/`SpanRecord`, isolates `span_001` per project run, retains `Sxxx` mapping, and preserves existing clearance/risk test expectations.
5. **Persistence/reopen tests:** create a project, confirm towers/spans, close/reopen, verify stable IDs/status/markers, source relocation behavior, and partial-run recovery.
6. **Viewer POC tests:** static cache manifest validation, Potree load/pick/marker focus in a controlled Windows WebView2 environment, no source path leak, and deliberately low LOD pick behavior.
7. **Performance tests:** separately benchmark small and representative large source/cache scenarios against agreed interaction and memory criteria. They are operator-controlled, not ordinary unit tests.
8. **Prediction exchange tests:** deterministic bundle IDs/manifests, result mapping by IDs not order, missing/unknown provider, mismatched source/run/schema rejection, traversal defense, and old-project compatibility.
9. **Manual acceptance:** execute the V1 Owner scenario in Section 29 using an approved representative source and record the viewer/cache/hardware versions.

## 27. Phased Implementation Plan

Each phase is independently reviewable. It must be authorized before implementation, and no phase authorizes a scientific algorithm redesign.

| Phase | Deliverable | Explicit boundary |
| --- | --- | --- |
| P1 | Project, Tower, Span, status, and prediction-envelope contracts | No viewer or pipeline change. |
| P2 | Minimal atomic project persistence/open/reopen | No cache conversion or desktop shell. |
| P3 | Potree proof of concept on small/existing sample | No full source conversion. |
| P4 | Full-source viewer-cache generation job/manifest | Separate from science; cancellable. |
| P5 | Viewer-pick to Python approximate-XYZ bridge | No new pole detector. |
| P6 | Existing local pole-refinement integration | Preserve diagnostics/failures. |
| P7 | Tower Registry markers/tree UI | No multi-span analysis yet. |
| P8 | Sequential Add Next Tower workflow | Linear topology only. |
| P9 | Span creation/topology/corridor overlay | No all-spans scientific algorithm. |
| P10 | Existing single-span pipeline adapter | Isolated per-span runs; preserve artifacts. |
| P11 | Per-span result navigation | No project aggregation calculations. |
| P12 | Project-level summaries | Read-only aggregation. |
| P13 | pywebview desktop shell | No package release. |
| P14 | Windows EXE packaging/installer validation | Static analysis still works without prediction. |

Prediction export/import should be introduced only after P1/P2 provide stable identities and P10 provides a validated static span result. It is an optional subsequent work item, not a hidden prerequisite for P13 or P14.

## 28. Open Questions

1. What representative Windows GPU and WebView2 runtime versions define the field-support baseline for the Potree POC?
2. Must the first desktop installer provision WebView2, or can it have a documented prerequisite with a diagnostic fallback?
3. Is the desired topology strictly a linear chain in V1, or should disconnected lines/branches be included before P9?
4. What is the approved source-relocation policy when a file is byte-different but a formal content-equivalence audit says it is equivalent?
5. Which HAG artifact is eligible for full-scene viewer coloring, and what provenance must its cache layer declare?
6. Where should large viewer caches live for field deployments: inside the project for portability, or in a machine cache keyed by source identity with project references?
7. What persistence/review policy is required when a confirmed tower changes after some adjacent spans have completed analysis?
8. What minimum user-visible performance targets (first load, navigation, pick/refine latency) and hardware envelope should qualify the full source cache?
9. What prediction bundle fields/artifacts will the independent prediction computer actually require once its design is available?

## 29. Acceptance Criteria

The future V1 is accepted only when an Owner can:

1. Launch TreeLineWarning and create a new project.
2. Select an existing LAS/LAZ and validate its identity.
3. Open a full-scene 3D LOD viewer and smoothly rotate, zoom, pan, and change the required views.
4. Click near a physical pole, review high-resolution refinement, and confirm persistent `T001`.
5. Confirm `T002`, observe `S001 = T001 -> T002`, and confirm `T003`, observing `S002 = T002 -> T003` without selecting `T002` again.
6. Select `S001` and independently run/open the existing scientific span workflow.
7. Select `S002` and independently run/open its workflow.
8. Close and reopen the application and see the same stable tower/span IDs, topology, source binding, and span statuses.
9. Use static analysis successfully with prediction absent and the UI showing `Prediction: NOT_CONNECTED`.
10. Retain access to the Legacy / Debug / Fallback Plotly selector during validation.

## Prediction Integration Boundary

Prediction is an optional downstream consumer of validated static project/span/tree results. The mandatory core remains:

```text
Source -> Ground/HAG -> Wire -> Vegetation -> Trees -> Clearance -> Static Risk
```

The optional extension begins only after a static analysis run is complete and identity-valid:

```text
Static analyzed project/span/tree state -> Prediction Adapter
  -> growth prediction -> future clearance -> future risk/warning results
```

No static stage waits for, imports, installs, or launches prediction. There are no placeholder predictions, default numerical estimates, ML dependencies, or EXE-startup requirements in V1.

## Prediction Provider Abstraction

Reserve names and a minimal versioned envelope; do not freeze model feature vectors.

```text
PredictionCapability
  provider_id/version (nullable while NOT_CONNECTED)
  supported input/result schema versions
  supported execution modes: file_exchange | local_api | subprocess | plugin | remote
  status and human-readable diagnostic

PredictionProvider
  describe_capability()
  export_request(request)
  validate_result(result)
  [later] execute(request)

PredictionRequest
  request_id, schema_version, project/source/tower/span/tree/run identities
  coordinate context, static artifact references/digests, timestamps
  model_specific_payload (namespaced, versioned, extensible)

PredictionResult
  result_id, schema_version, request binding, provider/model metadata
  identity-keyed result entries, artifact references/digests
  model_specific_payload (namespaced, versioned, extensible)

PredictionArtifactReference
  project-relative or bundle-relative path, role, SHA-256, byte size, schema/version
```

The core project validates the stable envelope and preserves/rejects unknown model-specific fields according to schema compatibility. It does not parse a provider’s internal model payload to decide static risk.

## Cross-Computer Prediction Exchange

The first integration uses a portable, deterministic, file-based exchange rather than a network service:

```text
TreeLineWarning project
  -> export validated prediction input bundle
  -> transfer to prediction computer
  -> prediction execution
  -> prediction result bundle
  -> validate and import into same project/span
```

Input bundles contain a canonical JSON manifest written last as completion marker, identity-keyed static metadata, only explicitly selected supporting artifacts, bundle-relative paths, SHA-256/size records, coordinate context, and a deterministic bundle ID. They must not duplicate the entire source LAS by default. Result bundles bind to request ID, project ID, source content identity, analysis run ID, span/tower/tree references, provider/model version, and input-bundle identity.

This mirrors the repository’s existing immutable, digest-checked cross-computer I1 pattern without declaring I1 itself to be the prediction interface. Later local API, subprocess, plugin, or remote provider modes may implement the same request/result envelope.

## Prediction Identity Mapping

Every request/result entry must map through stable IDs, never array ordering. At minimum each reference contains:

- `project_id`;
- `source_content_identity_digest` and relevant source record digest;
- `tower_id` and revision where applicable;
- `span_id` and geometry/tower-revision digest;
- `analysis_run_id` and static-result schema/version;
- `tree_id` / tree identity;
- coordinate frame, units, origin context/digest, and transform declaration; and
- request/result schema versions.

Current pipeline tree integers are span/run-local. The adapter must create immutable analysis-scoped tree IDs such as `<analysis-run-id>:tree:<local-tree-id>` and persist a map to the original tree artifact, stable nearest-source evidence where available, and the project `span_id`. A re-analysis creates a new `analysis_run_id` and new tree observations rather than mutating old prediction references. Cross-run biological identity matching is a future explicit capability, not an unstated assumption.

## Prediction Artifact Ownership

Reserve, but do not create, these paths:

```text
prediction/
  project_prediction_manifest.json
spans/S001/prediction/
  prediction_input_manifest.json
  prediction_result.json
  artifacts/
```

Input and result artifacts are owned by the project/span prediction area and referenced from the span result index. They cannot overwrite core static artifacts. A valid imported prediction result is an additional downstream artifact tied to one immutable static run; it becomes stale when its source static run is no longer current, but historical prediction results remain reviewable with their provider/model metadata.

## Prediction Capability / Failure States

The project and every span expose exactly these capability states:

| State | Meaning |
| --- | --- |
| `NOT_CONNECTED` | Default. No compatible provider is configured; no prediction exists. |
| `AVAILABLE` | A provider/bundle mechanism is discoverable, but a request is not yet eligible. |
| `READY` | Compatible provider and validated static inputs are available. |
| `RUNNING` | A provider job or exchange is in progress. |
| `COMPLETE` | A validated result bundle is imported and bound to a static run. |
| `FAILED` | Provider/export/import/validation failed; static analysis remains valid. |
| `INCOMPATIBLE` | Provider, schema, identity, coordinate context, or static run is incompatible. |

For this repository’s first project version, the expected honest state is `NOT_CONNECTED`. The UI shows the node, status, and an explanatory message; it must not hide the feature or render invented future warnings.

## Prediction Forward Compatibility

Core project loading tolerates old projects with no prediction field by materializing `NOT_CONNECTED` in memory. Newer prediction records are isolated in versioned envelopes: unknown optional fields are retained where safe, while an unknown required schema, incompatible source/run identity, or unsupported coordinate context produces `INCOMPATIBLE` without corrupting project/tower/span data.

Prediction is never allowed to redefine a Tower, Span, source identity, clearance result, static risk threshold, or static report. Its results are additive and explicitly versioned.
