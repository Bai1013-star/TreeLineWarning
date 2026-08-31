# DBH Recovery V1 acceptance — 2026-08-31

Baseline: `e37371718c64d4efee9472b1b99a933510875043`.
Project: `E:/2026.8.31-2/2026.08.31-2/2026.08.31-2`.
Run: `analysis-d434dfe375814b2d9b8a1d765a3e4d13`.
Design and public methodology/license review: `DBH_RECOVERY_V1_CONTRACT.md`.

## Result

Five recovered point-cloud DBHs pass V1 quality checks. Seven remain unavailable:
five have historical lasso provenance that cannot prove excluded stem ownership;
T-1-003 lacks accepted sections on both sides of 1.30 m; T-1-005 has no accepted
continuous breast-height column. This is not a claim that the raw scan contains
no stem. Raw queries found breast-height points for both T-1-005 and T-1-008.

Only five reliable measured/recovered trees are available. The local allometric
model requires ten plus leave-one-out validation, so it produces zero estimates.
No thresholds were relaxed to fill the table. HIGH describes internal geometry
quality; independent field measurements were not supplied for accuracy validation.

| Tree | DBH cm | Method | Quality | Fit evidence points | RMSE m | Accepted sections |
|---|---:|---|---|---:|---:|---:|
| T-1-001 | — | UNAVAILABLE | UNAVAILABLE | 0 | — | 0 |
| T-1-002 | 16.02123 | RECOVERED_STEM | HIGH | 39356 | 0.003911 | 4 |
| T-1-003 | — | UNAVAILABLE | UNAVAILABLE | 37240 | — | 3 |
| T-1-004 | 17.58796 | RECOVERED_STEM | HIGH | 31595 | 0.003895 | 4 |
| T-1-005 | — | UNAVAILABLE | UNAVAILABLE | 0 | — | 0 |
| T-1-006 | — | UNAVAILABLE | UNAVAILABLE | 0 | — | 0 |
| T-1-007 | — | UNAVAILABLE | UNAVAILABLE | 0 | — | 0 |
| T-1-008 | 18.06180 | RECOVERED_STEM | HIGH | 37196 | 0.003552 | 4 |
| T-1-009 | — | UNAVAILABLE | UNAVAILABLE | 0 | — | 0 |
| T-1-010 | 16.81874 | RECOVERED_STEM | HIGH | 27596 | 0.003767 | 6 |
| T-1-011 | — | UNAVAILABLE | UNAVAILABLE | 0 | — | 0 |
| T-1-012 | 16.11031 | RECOVERED_STEM | HIGH | 38547 | 0.003778 | 5 |

Raw recovery was attempted for seven trees and succeeded for five. The final
acceptance reused the hash-verified saved local-query evidence for all seven;
it executed zero new source/cache queries. Exact source original_index is retained
after independent vertical filtering. A source index cannot belong to two final
DBH trees; cross-span duplicate ownership fails publication closed.

## Preserved Survey authority

Tree count remains 12 and final membership remains 4,765,121 points. All 15
critical artifact hashes from the prior audit match, including existing final
membership and science artifacts. Persistent/global/display identities, height,
crown widths/area, clearance, danger counts and current risk compare exactly
against the original workbook. Ground, Wire, hybrid segmentation, reconciliation
and risk implementations were not edited. No 105M-point pipeline was run.

The authoritative DBH record is a separate finalized-geometry sidecar, not a
mutation of crown membership. Geometry generation:
`artifacts/project-runs/batches/analysis-d434dfe375814b2d9b8a1d765a3e4d13/dbh_v1/59b60a361e464918ac5dcaba4973eee1/finalized_geometry.json`.
SHA256: `4094a449ef1b2d57fae5ef0e7cc12520a5307e902b423764407ab9159c90401e`.

New Prediction workbook: `exports/Prediction_Input.DBH_RECOVERY_V1_03.xlsx`.
SHA256: `e8241716b5cd83c9654f6bec3b11937b8bf58f909aff34d970d9a6a482695069`.
The original `Prediction_Input.xlsx` is preserved. New DBH diagnostic columns are
appended; all 34 existing column positions remain unchanged.

## Validation and evidence

Evidence roots are preserved under `acceptance_evidence/`:

- `dbh_12_tree_audit_20260831`: original immutable audit.
- `dbh_recovery_v1_20260831`: initial primary-axis probe.
- `dbh_recovery_v1_20260831_02`: alternate validated stem column, exact raw ROIs,
  continuity probe, fresh desktop hydration and Survey export verification.
- `dbh_recovery_v1_20260831_03`: final geometry, seven exact query replays,
  final workbook/non-regression verification, and final fresh-process verification.

The first final-workbook assertion used exact floating equality and stopped on
Excel's 16-digit serialization (maximum difference 3.553e-15 cm). `verify.py`
validated the existing output with 1e-12 cm DBH-only absolute round-trip tolerance;
no recovery/export was repeated and all non-DBH comparisons remained exact.

Focused final command (68 passed, 7.32 s):

```text
.venv311/Scripts/python.exe -B -m pytest tests/manual_span/test_dbh_recovery.py tests/stage_b/test_dbh_recovery.py tests/stage_b/test_tree_metrics_export.py tests/stage_b/test_tree_display_export.py tests/stage_b/test_editor_clearance_invalidation.py tests/desktop/test_current_report.py tests/desktop/test_prediction_export_contract.py -q -p no:cacheprovider --basetemp C:/Windows/Temp/treeline-dbh-v1-final-01
```

Earlier related geometry/editor/clearance/Survey regression produced 82 passes
and one obsolete last-column assertion; that assertion was corrected to verify
the original display-ID position and appended DBH columns, then passed. No full
repository suite was run. Scoped Ruff, Python in-memory compilation, Node syntax
and git diff whitespace checks pass. Tests cover robust fit, partial circumference,
continuity/clutter, editor delete/lasso/merge/undo, same-count journal changes,
source/artifact tampering, duplicate stem ownership, local raw recovery, labels,
backward compatibility and DBH failure isolation from successful finalization.

Final fresh-process desktop/API and Survey export: PASS, 175.82 s. The actual
`DesktopWorkflow.current_report()` hydrated all twelve finalized trees and matched
all numeric DBH values, methods, quality, reasons and display identities. Only
application-log writes were redirected to the audit; no science path was mocked.
All preexisting project file sizes and modification times remained unchanged.
Receipt: `dbh_recovery_v1_20260831_03/restart.json`.
New Survey workbook: `exports/Survey.DBH_RECOVERY_V1_03.xlsx`, SHA256
`60b88f80619de55f641463f010227cde21368f06dfe1acac37b389c0f78544d1`.
UI source labels are verified by executable JavaScript tests, not a new packaged
GUI session. No package rebuild or push is part of this DBH-only gate.

## Git scope

DBH modules, finalize/export/report adapters, formatter, focused tests and these
two documents only. Existing `core/stage_b/batch.py` changes (82 additions,
22 deletions) and all unrelated untracked work remain excluded and preserved.
Parent review and the DBH-only implementation/12-tree/non-regression gates PASS.
This authorizes a scoped local commit, not packaged or clean-PC release acceptance.
