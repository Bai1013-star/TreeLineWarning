# Survey V0.1 REAL Multi-Span E2E Gate

Status: **DESIGN-FROZEN / EXECUTION-PENDING**

Gate ID: `SURVEY_V01_GATE_B_REAL_MULTISPAN_E2E`

## 1. Purpose

This gate answers one product question only:

> Can the current Survey V0.1 product process one real point-cloud source across at least two adjacent real logical Spans, preserve global tree ownership, complete post-editor clearance/export, and resume from persisted state without substituting fixture science?

This is an acceptance gate, not an algorithm redesign task. Existing validated Ground, Wire V2, Vegetation/FRV, `CROWN_STEM_HYBRID`, global reconciliation, Tree Editor, Clearance, Risk, and Excel semantics remain authoritative unless the gate exposes a reproducible defect.

## 2. Required test scene

The acceptance scene MUST satisfy all of the following:

- One real LAS or LAZ source readable by the production runtime.
- At least three confirmed Towers in one ordered Line, producing at least two adjacent logical Spans: `Tower A -> Span 1 -> Tower B -> Span 2 -> Tower C`.
- The selected Spans MUST use the same source identity and shared-source preparation/cache path.
- The scene SHOULD contain vegetation in both Spans and SHOULD contain at least one tree near the shared Tower boundary so cross-Span ownership can be inspected.
- The first execution SHOULD use manual Tower confirmation when possible. GIS registration may be tested separately so a GIS defect does not hide a multi-Span science defect.
- Display/LOD samples MUST NOT be used as science membership authority.

## 3. Frozen production path

The run MUST use `mode="REAL"` through the desktop/product workflow.

The REAL path MUST NOT inject:

- `_fixture_runner`
- `_fixture_preparer`
- synthetic tree outcomes
- fixture-only Ground/Wire/Vegetation evidence

Expected science path:

`Project/Scan -> Shared Source Preparation -> Ground -> Wire V2 -> Vegetation/FRV -> CROWN_STEM_HYBRID -> per-Span results -> global reconciliation -> Tree Editor -> final clearance -> risk publication -> Prediction_Input.xlsx`

## 4. Pre-run checks

Record before execution:

- repository branch and HEAD
- Python/runtime identity
- source path, source size, source SHA-256/source identity, and point count
- coordinate declaration and confirmed linear unit
- Tower IDs, ordered Tower names, and confirmed XYZ
- selected Span IDs and endpoint Tower IDs
- current risk threshold
- whether source/shared caches already exist (`COLD` or `REUSE`)

The run MUST fail closed if the source, unit, Tower endpoints, registration state, or required science backend is not valid.

## 5. Execution sequence

### Gate B1 — Project and topology

1. Create/open a real Survey project.
2. Import the real LAS/LAZ source.
3. Confirm the linear unit explicitly when required.
4. Confirm at least three ordered Towers.
5. Verify at least two adjacent logical Spans exist and reference the shared source rather than copied full-resolution files.

PASS requires:

- one authoritative Scan/source identity
- `span_count >= 2`
- every selected Span has valid endpoint Towers
- no full-resolution point-cloud copy per Span

### Gate B2 — REAL batch science

Start one batch containing at least two adjacent Spans with `mode="REAL"`.

PASS requires:

- acceptance mode is `REAL`
- source preparation succeeds
- at least two selected Spans enter production science
- no selected Span result contains fixture-only acceptance evidence
- Ground, Wire, Vegetation/FRV, and Tree stages use the production implementations
- Tree backend is `CROWN_STEM_HYBRID`
- failures, if any, are isolated and surfaced with stage/code/message rather than silently replaced with fixture output

### Gate B3 — Shared-source and performance invariants

PASS requires:

- one source identity is shared by the selected Spans
- source-level preparation/cache is reusable across the batch
- Span-local science does not require a full copy of the raw point cloud
- telemetry is persisted for source preparation, Span count, Span-local reads, cache hit/miss, total time, per-Span time, and stage time when available
- a second run/restart MUST reuse valid completed work/caches where the existing contract permits reuse

Performance numbers are evidence, not a hard machine-specific threshold. No RTX/CPU/RAM model may be hard-coded into acceptance logic.

## 6. Global tree reconciliation acceptance

After automatic science completes, inspect the global reconciled tree set.

PASS requires:

- `duplicate_final_original_index == 0`
- one original point is not owned by two final trees
- cross-Span halo/context observations do not create duplicate physical trees
- a tree near the shared Tower/Span boundary has one stable final owner
- changing Span processing order does not change final ownership semantics
- persistent tree identity is available for downstream restart/export logic

Any violation of exact membership conservation or exclusive final ownership is a **BLOCKER**.

## 7. Tree Editor acceptance

Using the real reconciled result:

1. Open Tree Review.
2. Perform at least one supported real edit (`DELETE` or `MERGE`).
3. Perform `UNDO` at least once.
4. Finalize a post-editor tree revision.

PASS requires:

- edit operations act on exact final membership, not display-only samples
- DELETE/MERGE changes the intended final tree membership only
- UNDO restores the previous revision
- editor operations are persisted
- closing and reopening the project restores the same editor revision/history

## 8. Clearance and risk acceptance

Run final clearance from the finalized post-editor tree set.

PASS requires:

- clearance is calculated from final post-editor tree membership and reconstructed Wire evidence
- nearest tree/wire evidence is persisted for valid records
- `WARNING` is produced only for non-contact positive clearance within the configured threshold
- `SAFE` is produced above the configured threshold
- zero/tie-level clearance MUST NOT be promoted to `DANGER` merely because the numeric distance is zero
- until the reviewed physical CONTACT detector exists, zero/tie-level cases remain `CONTACT_UNCERTAIN / REVIEW_REQUIRED`

The absence of the final physical CONTACT detector is a known V0.1 product blocker and is reported separately from this multi-Span execution gate.

## 9. Excel / Prediction interface acceptance

Finalize and export `Prediction_Input.xlsx`.

PASS requires each final exported tree row to remain bound to the finalized global tree set and, where available, include:

- `persistent_tree_id`
- display/analysis tree identity
- project/line/span identity
- start/end Tower identity
- tree position
- tree height
- crown width metrics
- DBH/diameter when valid Ground-bound measurement is available
- minimum clearance
- current risk
- contact state/evidence
- review/edit provenance
- warning threshold used

The exporter MUST fail closed instead of exporting pre-editor or non-finalized candidate membership as final prediction input.

## 10. Restart acceptance

After a successful finalized run:

1. Close the application/process.
2. Reopen the same project.
3. Re-enter the project without rerunning completed science manually.

PASS requires:

- project topology is restored
- previous batch/run identity is restored
- completed Span state is restored
- Tree Editor operations/final revision are restored
- final tree set remains consistent
- final clearance/export state remains attributable to the restored editor revision
- a completed Span is not unnecessarily rerun simply because the application restarted

## 11. Required evidence bundle

Create one acceptance report containing at least:

```text
GATE_ID = SURVEY_V01_GATE_B_REAL_MULTISPAN_E2E
RESULT = PASS | FAIL | BLOCKED
BRANCH = ...
HEAD = ...
RUNTIME = ...
SOURCE = ...
SOURCE_SHA256 = ...
SOURCE_POINT_COUNT = ...
TOWER_COUNT = ...
SPAN_COUNT = ...
SELECTED_SPANS = ...
ACCEPTANCE_MODE = REAL
SOURCE_CACHE_HINT = COLD | REUSE
SOURCE_PREP_COUNT = ...
GROUND_PREP_COUNT = ...
SPAN_LOCAL_READ_COUNT = ...
CACHE_HIT_COUNT = ...
CACHE_MISS_COUNT = ...
TOTAL_TIME = ...
STAGE_TIME = ...
FINAL_TREE_COUNT = ...
DUPLICATE_FINAL_ORIGINAL_INDEX = 0
EDITOR_OPERATIONS = ...
RESTART_STATE = ...
FINAL_CLEARANCE_STATUS = ...
EXCEL_PATH = ...
CONTACT_DETECTOR = PENDING_REVIEWED_PHYSICAL_DETECTOR
BLOCKERS = ...
```

Also retain the project-local authoritative artifacts/logs needed to reproduce a failure. Large raw LAS/LAZ files do not need to be committed to Git.

## 12. PASS / FAIL / BLOCKED rules

### PASS

`PASS` requires all of the following:

- real LAS/LAZ source
- at least two adjacent Spans
- `mode=REAL`
- production science used
- selected Spans complete or only non-critical, explicitly accepted review states remain
- exact global reconciliation invariant holds
- real Tree Editor operation and restart persistence pass
- final clearance runs on the post-editor revision
- final Excel export succeeds
- no silent fallback to fixture/synthetic science

### FAIL

Use `FAIL` for a reproducible product/science defect in code that prevents one of the required acceptance conditions.

### BLOCKED

Use `BLOCKED` only when execution cannot fairly test the product because of an external acceptance prerequisite, for example:

- no suitable real multi-Span point-cloud scene is available
- required local runtime dependency is not installed on the acceptance machine
- the source file is unreadable/corrupt

A code defect MUST NOT be relabelled as `BLOCKED`.

## 13. Failure triage order

When the gate fails, fix only the first authoritative failing layer:

1. source/capability/unit
2. Tower/Span topology
3. shared source preparation / Ground
4. Wire
5. Vegetation/FRV
6. Tree segmentation
7. global reconciliation
8. Tree Editor persistence
9. final clearance/risk
10. Excel export
11. restart/desktop publication

Do not change downstream algorithms to compensate for an upstream contract failure.

## 14. Gate closure rule

This gate is closed only after one real scene produces a reproducible evidence bundle with `RESULT = PASS` on `feature/fast-research-pipeline` (or its designated successor branch).

After this gate passes, the next primary science blocker is the reviewed physical `CONTACT -> DANGER` detector, followed by final release/cross-machine acceptance.
