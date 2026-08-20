# TreeLineWarning V0.1 Master Plan

## Stage A — Product foundation (this delivery)

Stage A is implemented as a sidecar product layer in `core.stage_a` so the
validated P1/P2 project contracts and old single-span workspaces remain
unchanged.  It provides:

1. Stable Project/Line/Tower/Span/Scan/AnalysisRun records and strict JSON
   round-trip persistence.
2. Capability reports with explicit unknown/confirmation states.
3. CRS/unit safety and metric-distance admission rules.
4. Canonical GIS adapters for KML and CSV.
5. Registration states and one/two-anchor rules.
6. Shared source preparation and bounded local access metrics.
7. Versioned RiskPolicy and independent review state.
8. Fully offline runtime contract (`RUNTIME_NETWORK_REQUIRED = NO`): local
   project lifecycle, local GIS/point-cloud processing, vendored Viewer assets,
   local Plotly assets, and loopback-only API/static serving. V0.1 uses
   `NO_BASEMAP`/`OFFLINE_BASEMAP` only; no online basemap provider exists.
9. Line/Route topology foundation with main/branch relations, ordered Tower
   sequences, adjacent logical Span generation, and an explicit shared-source
   local-access partition policy. This is topology/UI foundation only; it does
   not implement Stage B batch science.

The current single-span scientific workflow remains authoritative.  Stage A
does not rerun Ground/Wire/Vegetation/Tree artifacts merely to create the
product records.

## Stage B — Multi-Span Science Productization

1. Connect `Project` topology and `Scan` identity to the existing
   source-preparation/tile-cache manifests.
2. Replace per-span orchestration with one source-level preparation and
   bounded Span queries through `SharedSourceAccess`/the existing role tile
   cache.
3. Resolve real GIS field format and implement the required adapter and
   registration review UI.
4. Map canonical Tower/Span records to the existing ownership semantics,
   preserving APEX-IN-SPAN-CORE ownership and exclusive tree allocation.
5. Productize multi-span Ground/Wire/FRV/CROWN_STEM_HYBRID execution without
   changing validated scientific semantics.
6. Define and validate CONTACT evidence using reconstructed continuous Wire
   geometry and Tree evidence; only then bind it to RiskPolicy.
7. Add multi-span regression, real-data artifact provenance, and performance
   gates.

## Stage C — Desktop Product Closure

1. Complete native project open/save/import and registration review flows.
2. Harden Viewer/Tree Editor/export workflows for persisted multi-span runs.
3. Add packaging, offline diagnostics, installation tests, and final EXE
   acceptance. Any future basemap must be local/offline data; an
   `OnlineBasemapProvider` is out of scope.

## Full-offline acceptance gate

`RUNTIME_NETWORK_REQUIRED = NO` is a release gate. Runtime-required
JavaScript, CSS, Plotly, LAZ decompression, and point-cloud data are packaged
or served from the project/application directory. Network-looking URLs in
tests or documentation are not runtime dependencies; the application may use
`127.0.0.1`/`localhost` for its own loopback server only. Acceptance tests
reject CDN/remote-font/remote-tile/remote-API references in runtime surfaces
and reject `ONLINE_BASEMAP` in a Stage-A project.

## Explicit non-goals

PPK solving, ROS bags, camera projection, species models, blind tower
detection, cloud services, advanced mesh, Tree brush/polygon/manual split,
final CONTACT implementation, and final PyInstaller packaging remain outside
Stage A.
