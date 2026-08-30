# REAL finalized export and restart acceptance

Accepted on 2026-08-30 by the supervising review, continuing the existing REAL
multi-Span evidence on `27f42e19b5206a5912613f199165347ba59bdd4c` (which includes
`c32cc489`). The product export repair is commit `f406c37` on
`codex/final-deliverable-20260830`. This is a continuation of the approved REAL
analysis, not a replacement by a fixture or a fresh segmentation claim.

## Authoritative result

- Project: `workspace/survey_v01_gate_b_20260830`.
- Run: `analysis-53f4c1ac401249d78c7e8a25a438d9c9`.
- Prior REAL evidence establishes forward/reverse lineage and clearance
  equivalence, editor DELETE/UNDO, final DELETE and explicit Finalize.
- Final inventory: 16 persistent trees, global IDs 1–16, 3,750,317 exact members,
  zero duplicate ownership, one persisted editor operation.
- Current clearance: VALID, 16 records, 72 edges, minimum
  4.543721626531224 m. Workbook current risks are SAFE. No new CONTACT evidence or
  DBH measurement is claimed.

## Default export: PASS

The known failed workbook (12,273 bytes, SHA256
`4586f52e97a54d660bbd7f614a18a1d44fb837bed4e5e47df1b2621a806417f1`)
was copied to an exclusive evidence backup and left in the product export path.
The repaired default `DesktopWorkflow.export_prediction()` invalidated this
cache and replaced it without identity overrides, manual cell filling, implicit
Finalize, or a science rerun.

The replacement has 16 rows, correct primary-Span Tower endpoints in all 32
endpoint cells, 16 true blank DBH cells with the unavailable reason, current
run attribution and no future fields. Independent supervisor comparison found
that exactly the 32 formerly blank endpoint cells changed; all other tree cells
are identical. Replacement size: 12,387 bytes; SHA256:
`cf3b5c0c88c3abaa0d4314c38944e31daef93e5d4e0cfaadb9126f79fe86d717`.

## Fresh-process restart: PASS

A separate Python process reopened the project and restored the same run,
finalized Span statuses, 16 final records, 3,750,317 exact members, one editor
operation, current clearance and workbook. It did not rerun science, apply the
edit twice, call Finalize, or regenerate the workbook.

Before/after manifests covering the raw LAS, project files, authoritative batch
directory and shared cache are equal for both export and restart. The common
manifest SHA256 is
`bc21bd33d5dfef8da596de9b96e9e014c8e318257aada84b1c98b73944854186`.
Earlier evidence is unchanged. Only expected append-only application log events
were permitted. Raw LAS SHA256 remains
`6d7ae65ff91d22e62ba97321b975949adf9e361596b733d7c799574877aae596`.

## Evidence and limits

Evidence is under `acceptance_evidence/final_delivery_20260830`:

- `continuation_export_acceptance.json` and before/after authority manifests.
- `sol_export_independent_semantics.json`.
- `continuation_restart_acceptance.json` and fresh-process transcript.
- `continuation_backup_failed_export/Prediction_Input.failed.xlsx`.
- `continuation_export_restart_acceptance.py` and its runbook.

The first harness invocation stopped before any product mutation because its
legacy evidence scanner followed an external dependency junction. The scanner
was corrected to record reparse entries without descending into them; the
original failure log is preserved. The second export invocation passed.

Independent focused review: 78 tests passed, plus seven export compatibility
tests after the final correction; changed-file Ruff passed. The initial broad
suite is not wholly green and is being separately classified; this document
does not claim full-regression, packaged-runtime or clean-PC acceptance.

The dashboard's old per-Span summary sum is 8 + 7 = 15 while the finalized
global count is 16. This is an identified presentation mismatch, not lost
membership. Modern current-report display wiring also remains under separate
repair/visual acceptance. Parallelism, future Preview and external delivery are
not accepted by this gate continuation.
