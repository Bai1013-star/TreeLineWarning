# Windows Survey baseline02 acceptance — 2026-08-31

Status: **PASS on the current Windows host**, including relocation to a path containing Chinese characters and spaces. This closes the baseline package gate before parallel execution and local Future Preview implementation. It does not claim clean-PC acceptance or legal clearance.

Build source: `c4f78396363a40db1a204c25e32289b8e3912ff7`, branch `codex/final-deliverable-20260830`. The build receipt records identical start/end source HEAD and tracked digest `2e52d37502c811e8503b77327616742d43cec4e9a35d5f1fb70d3d260435f3b9`.

Package: `dist/TreeLineWarning-survey-baseline-20260831-02/TreeLineWarning/TreeLineWarning.exe`. Accepted relocated copy: `dist/交付验收 含空格/TreeLineWarning Survey 基线02/TreeLineWarning.exe`.

EXE: 22,918,966 bytes, SHA256 `ac6de06c9d4b7d927baf7f758701f87747dd3a9c29d28351a1806d3e67696021`. All 606 manifest payload rows (706,685,318 bytes) match source and destination. There are 607 regular files including the self manifest; no reparse paths were copied.

## Verified behavior

- Frozen native dependency health passes from both original and relocated paths, including PDAL Ground/HAG, numeric EPSG reprojection, LAZ/LAS, Pillow, package-contained module/resource origins and WebView2 loader. Native PROJ and GDAL explicitly receive the UTF-8 data path; pyproj retains its own database.
- The relocated native window rendered the Chinese Survey home using EdgeChromium, showed no open project, and closed normally (exit 0). The parent inspected its screenshot and accessibility tree. Only the newly launched window was closed; other user applications were untouched.
- Two fresh frozen processes rehydrated the accepted REAL finalized run `analysis-53f4c1ac401249d78c7e8a25a438d9c9`. Reports match the approved baseline exactly, including tree attribution, W1 geometry, metrics, gate and workbook mapping. Missing project arguments fail with a diagnostic and nonzero status.
- The parent inspected the relocated backend's current report: 16 final trees, 3,750,317 member points, minimum clearance 4.543721626531224 m (display 4.544), 16 SAFE rows, 2 REVIEW_REQUIRED quality rows, 16 unavailable/blank DBH values, 30,000 displayed tree points and one fitted current W1 segment.
- Raw source, finalized authority, workbook, approved report baseline and legacy evidence remain unchanged. Authority digest before/after is `bc21bd33d5dfef8da596de9b96e9e014c8e318257aada84b1c98b73944854186`. Application log retains its original prefix and appends only 17 allowed read-only lifecycle/HTTP events (2,208 bytes). No analysis, editing, finalize or export was performed during this package acceptance.

## Evidence

All paths below are under `acceptance_evidence/final_delivery_20260830/`:

- `baseline_build02_20260831_console.log` and `baseline_build02_20260831_audit_relocation.{json,md}`; build receipt under `build/TreeLineWarning-survey-baseline-20260831-02/`.
- `packaged_baseline02_20260831_preflight/` for original-path dependency health.
- `packaged_baseline02_20260831_gui/` for native lifecycle, process result, screenshot, accessibility tree and parent visual acceptance.
- `packaged_acceptance_baseline02_20260831_relocated/` for dependency health, two survey probes, workbook attribution, invalid invocation, finite backend lifecycle, DOM/screenshots, parent browser acceptance and read-only guards. The full-page screenshot tool repeated a bottom region; the separate viewport screenshot and DOM confirm the actual page.
- `sol_unicode_runtime_review_20260831/` records the independent source validation: 71 passed, 1 skipped; scoped Ruff and diff checks passed. The skipped direct PDAL import uses the lean venv; explicit staged native child probes passed.

Baseline01 artifacts and its Unicode-path native failure are preserved. Baseline02 supersedes that runtime failure through a new source commit and new package/evidence; old results were not relabeled.

## Remaining release limits

Clean PC: **NOT_TESTED**. The verified offline WebView2 installer is bundled but was not executed. No Python/Conda/developer installation was required for the frozen probes. Third-party notices remain final_native03, with 31 explicitly unresolved provenance/legal-review entries including one unavailable local SQLite license text. Optional PDAL meshio/geopandas adapters are outside the accepted core scope. Final parallel/Preview source changes require a new final package and acceptance; this baseline is preserved unchanged.
