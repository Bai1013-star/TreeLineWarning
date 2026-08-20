# TreeLineWarning Survey Software V0.1 Requirements

## Scope

Stage A adds a product foundation around the locally validated single-span
workflow.  It does not replace the existing science or claim that the new
multi-span execution is complete.

## V0.1 MUST

- Persist stable opaque Project, Line, Tower, Span, Scan, and AnalysisRun
  identities in `stage-a-project.json`.
- Keep existing P1/P2 `project.json` workspaces and single-span artifacts
  readable through an explicit legacy-compatibility path; migration is never
  implicit.
- Report point-cloud capabilities as `AVAILABLE`, `NOT_AVAILABLE`, `UNKNOWN`,
  or `REQUIRES_CONFIRMATION` without vendor-specific branching.
- Keep horizontal CRS, vertical reference, coordinate kind, and linear unit
  explicit.  Unknown units cannot be used as metres; an explicit local metric
  declaration is allowed.
- Convert KML/CSV through canonical `TowerSeed` and `RouteSeed` adapters.
  CSV CRS and units are explicit inputs; numeric ranges are never used to guess
  a CRS.  Real field GIS acceptance remains pending until a field file arrives.
- Represent registration as a state machine.  One anchor assists matching;
  two anchors may establish a local 2-D rigid transform only when scale is
  consistent.  No-GIS manual tower selection remains available.
- Represent Line/Route topology with explicit `MAIN`/`BRANCH` roles, parent
  relationships, and ordered Tower sequences.  Stage A may generate adjacent
  logical Spans from those sequences for UI/topology preview; this does not
  start batch science processing.
- Provide `SharedSourceAccess` with one source preparation pass, a reusable
  chunk spatial index, and bounded local LAS/LAZ reads for future Span loops.
- Keep each logical Span tied to shared source/cache access; a Span must not
  require a copied full-resolution point-cloud file.
- Keep risk policy configurable with `warning_clearance_threshold_m=0.5` by
  default.  Risk (`DANGER`, `WARNING`, `SAFE`) and quality/review
  (`VALID`, `REVIEW_REQUIRED`) are separate.
- Run fully offline: `RUNTIME_NETWORK_REQUIRED = NO`. Project creation/opening,
  LAS/LAZ import and processing, KML/CSV/GIS import, local Tower/Span
  visualization, the existing point-cloud Viewer, Ground/Wire/Vegetation/Tree/
  Tree Editor/Clearance/Risk, and Excel export must not contact the Internet.
  Loopback HTTP is allowed only for the application's own local API/static
  server. Google Maps, OpenStreetMap, Mapbox, ArcGIS Online, remote fonts,
  CDN JavaScript/CSS, telemetry, update checks, and remote APIs are forbidden
  runtime dependencies.
- V0.1 map display is a local coordinate/schematic canvas containing imported
  Line/Route, Tower, Span, and Risk/Analysis status. It has no geographic
  background map. Basemap policy is bounded to `NO_BASEMAP` or
  `OFFLINE_BASEMAP`; `OnlineBasemapProvider` is not implemented or required.
- Keep the existing Viewer data boundary: full-resolution original-index
  membership remains science data; display samples/LOD remain viewer data.

## V0.1 OPTIONAL

- Minimal KML and CSV adapters (implemented when dependencies remain standard
  library only).
- A richer basemap may be added later only from packaged or explicitly
  imported offline data.
- Desktop shell actions for New/Open Project, point-cloud capability display,
  and GIS import.
- Persisting a shared source index to a project-local cache directory.

## FUTURE

Stage A intentionally does not implement PPK/ROS, camera projection, tree
species recognition, blind tower detection, cloud/multi-user storage, advanced
mesh, Tree brush/polygon/manual split, final CONTACT science, or final EXE
packaging.  COPC is an optimization option, not a required input format.

## Locked current science baseline

The latest local folder workflow is wired by
`algorithms.folder_pipeline.orchestrator.FolderOrchestrator` and reuses:

- Ground: `algorithms.vegetation.fast_research.run_fast_ground_hag`, with
  source-identity-checked cache reuse or a PDAL preflight.
- Wire: `algorithms.spans.wire.run_span_wire_pipeline` (authoritative Wire V2
  confirmation); vegetation exclusion uses only its final exact confirmed
  `source_index` membership.
- Vegetation: the high-recall Span-local region from the role-specific source
  tile cache and the existing FRV/original-index path.
- Trees: `algorithms.spans.tree_runner.run_span_tree_pipeline` with
  `FastBaselineConfig.tree_backend == "crown_stem_hybrid"`; legacy watershed
  output is retained only as crown-candidate/reference input, followed by
  stem-guided validation and full-resolution membership assembly.
- Tree Editor: `algorithms.folder_pipeline.tree_editor.TreeEditSession`,
  retaining DELETE/MERGE/UNDO and the original/final membership artifacts.
- Viewer: `algorithms.folder_pipeline.server` plus the existing Step-1/2/3
  HTML and viewer cache.  It does not materialize the full high-resolution tree
  membership for normal display.
- Clearance: `algorithms.spans.clearance.run_span_tree_wire_clearance` and the
  exact post-edit route used by the folder orchestrator.  Stage A defines a
  product RiskPolicy but does not implement the final CONTACT detector.
