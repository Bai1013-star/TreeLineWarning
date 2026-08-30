# Current finalized Survey report acceptance

The modern desktop current-report page passed independent REAL read-only
acceptance on 2026-08-30. This closes the blank KPI/3D presentation defect; it
does not claim a completed Windows package, full regression, parallel analysis,
or future Preview.

The accepted project is `workspace/survey_v01_gate_b_20260830`, finalized run
`analysis-53f4c1ac401249d78c7e8a25a438d9c9`.

## Observed result

- 16 final tree rows and 3,750,317 exact final member points.
- Global minimum clearance: 4.543721626531224 m; all 16 current risk values SAFE.
- Both existing REVIEW_REQUIRED quality flags remain visible. DBH is unavailable
  and remains blank rather than zero.
- 30,000 bounded final tree display points and 73 persisted W1 curve points in
  one fitted current-wire segment. The report does not use tower schematic lines.
- 160 values across ten fields match the accepted Prediction_Input workbook,
  including dimensions, clearance, risk, quality, display IDs and DBH metadata.
- The project dashboard shows FINALIZED and the authoritative global count 16,
  rather than summing the old per-Span diagnostic counts 8 + 7.

Browser screenshots and DOM snapshots confirm populated KPIs, the 3D point/curve
plot, 16 rows, current-only labels and the completed project dashboard.

## Read-only and finalization boundary

The gate verifies the persisted batch publication and hydrated result separately:
matching run identity and selected Span set, COMPLETE status, every selected Span
FINALIZED, global reconciliation, current journal revision and publication time,
and no active or failed finalization. Persisted finalization lifecycle survives
restart. Missing/corrupt publication or journal evidence fails closed.

The report requests a projection with optional danger-display persistence
disabled. Existing editor behavior retains its previous default. Report fields
are explicitly limited to current Survey values, with no forecast fields.

The independent service opened the existing project and read the report; it did
not analyze, edit, finalize or export. Before/after source and authoritative
project manifests were identical, with canonical digest:

`bc21bd33d5dfef8da596de9b96e9e014c8e318257aada84b1c98b73944854186`

The existing Prediction_Input workbook was also unchanged. The service and its
browser tab were closed after inspection.

## Validation and evidence

- Sol independent focused tests: **48 passed** across report, export, review,
  numbering, Stage-C workflow and desktop shell tests.
- Ruff passed for changed report/workflow/server/wrapper Python files and tests.
- Node syntax checks passed for the report and desktop JavaScript.
- Worker scoped run: **39 passed**.

Evidence under `acceptance_evidence/final_delivery_20260830/`:

- `sol_current_report_tests.log`
- `current_report_real_console.log`
- `current_report_real_acceptance/report.json`
- `current_report_real_acceptance/excel_report_semantics.json`
- `current_report_real_acceptance/read_only_guard.json`
- `current_report_real_acceptance/report.png` and `report_full.png`
- `current_report_real_acceptance/report_dom.txt`
- `current_report_real_acceptance/dashboard.png` and `dashboard_dom.txt`
- `current_report_real_acceptance/batch_ui.json` and `workspace_ui.json`

The retry adapter now passes the complete original selected Span set in topology
order. Its production cache/retry repair has a separate acceptance boundary and
must not be inferred complete from this report-only acceptance.
