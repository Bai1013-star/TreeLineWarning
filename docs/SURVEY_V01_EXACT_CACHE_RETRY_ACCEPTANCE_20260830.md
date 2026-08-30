# Stage-B exact cache and retry acceptance

This is bounded engineering acceptance on `codex/final-deliverable-20260830`,
reviewed by Sol High. It does not claim a new REAL science run, a Windows
package acceptance, parallel execution, or changes to frozen algorithms.

## Defect and repair

The old retry adapter selected only failed Spans, and the scheduler reused
successful Spans from display summaries that contained no tree or wire arrays.
Consequently a retry could report COMPLETE while omitting successful Spans
from global reconciliation. Re-entering an existing run could also rewrite
editor artifacts. The adapter repair is in commit `1aaf582`; this slice closes
the scheduler's exact-artifact contract.

Each successful native Span now publishes a separate cache index that binds
run, project, scan, product/internal Span, source identity, current source stat
(and content hash when configured), configuration, and shared preparation.
Tree and W1 files are hashed and must match their required project-local paths.
The scheduler reloads them through the existing tree and wire loaders. Missing,
legacy summary-only, mismatched, or damaged entries are cache misses. Custom
test runners that do not publish exact native artifacts remain non-authoritative
and are recomputed on restart.

Mixed cached and recomputed results enter global reconciliation and clearance
in canonical topology order. Retry preserves all originally selected Spans;
unselected project Spans remain outside the run.

A clean COMPLETE restart additionally verifies a detached proof of the batch,
global membership, reconciliation, clearance, and available exact danger-point
sidecars. It returns persisted results without source preparation or writes.
Near-wire points are bound to current original_index, tree ID, XYZ, and
persistent tree ID. NumPy indexing avoids a Python dictionary for every source
point. Missing required point evidence forces the normal recomputation path.

The scheduler rejects same-run FINALIZED state, any edit operations, malformed
edit journals, and invalid batch manifests before source preparation or writes.
The desktop workflow remains the supported route for reopening reviewed or
finalized projects.

## Independent validation

- Original whole-workflow reproduction: PARTIAL_FAILURE followed by retry
  previously retained only the failed Span; the repaired reproduction retains
  both Spans, two final trees, and all six exact fixture indices.
- All Stage-B tests plus the current-report suite: **93 passed in 9.83 s**,
  with the explicitly selected native runtime and a fresh system TEMP directory.
- After the final persistent-ID validation and allocation reduction:
  exact-cache plus Stage-B integration tests: **36 passed in 8.29 s**.
- Changed-file Ruff passed. No golden fixtures, source LAS, accepted REAL
  project, or old acceptance evidence were modified by this slice.

Tests cover two/three Spans, reversed request order, one failed Span, fresh
all-success equivalence, exact original_index/XYZ/tree IDs/lineage, W1 arrays,
clearance, missing/tampered cache files and binding/preparation/configuration,
near-wire point evidence, and immutable edited/finalized/malformed states.
No-write assertions compare file bytes and modification times.

Evidence is under `acceptance_evidence/final_delivery_20260830/`:

- `sol_retry_loss_reproduction.json`
- `sol_retry_loss_repaired.json`
- `sol_stage_b_cache_closed_tests.log`
- `sol_stage_b_cache_pid_tests.log`

## Scope limits

The frozen wire loader's strand display-offset behavior is unchanged; W1
arrays and deterministic clearance are the authoritative cache contract.
The known REAL project remains finalized and was deliberately not rerun through
this scheduler. Its separate export/restart/current-report acceptance still
applies. These tests do not replace the pending settled-tree full regression
or frozen application acceptance.
