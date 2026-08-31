# V4 implementation acceptance — ready for human review

Baseline: `09e07a56ef75274abf5a179080db2bfcf9f2a1dc`.
Authoritative project: `E:/2026.8.31-2/2026.08.31-2/2026.08.31-2`.
Evidence directory: `acceptance_evidence/dbh_recovery_v4_20260831`.

**12 finalized trees; 7 existing automatic DBHs; 5 still unavailable.**
No real stem choice or DBH fit was performed. These are review-ready sessions,
not five new scientific recoveries.

| Tree | Review mode | Candidates | Displayed exact raw rows |
|---|---|---:|---:|
| T-1-001 | Click raw stem seed | 0 | 80,000 |
| T-1-006 | Confirm candidate or seed | 3 | 80,000 |
| T-1-007 | Confirm candidate or seed | 2 | 80,000 |
| T-1-009 | Confirm candidate or seed | 2 | 80,000 |
| T-1-011 | Click raw stem seed | 0 | 9,220 |

`review_ready.json` contains session IDs, exact project paths and artifact hashes.
Seven V2 cached ROI captures were hash-checked and replayed, with zero new raw
source queries and no 105-million-point science run. Preparation took 111.66 s.
Every existing project file was preserved after closure. All critical science
hashes, twelve identities, 4,765,121 final members, seven DBH rows/evidence,
height/crown metrics, clearance/risk and old exports remain unchanged.

## Tests and parent review

The final selected regression is **169 passed in 25.63 s**. It covers 27 V4
test cases (20 backend, 3 live HTTP/API and 4 UI tests) plus V1/V2/V3 recovery,
current report, editor, lasso, clearance, labels and exports. The new tests include
synthetic candidate/seed confirmation, tilt, ambiguous neighboring stems,
missing BH geometry, invalid/deleted seeds, revision changes including merge/undo,
seven exact old measurements/arrays, separate-process restoration, and both
Survey and Prediction_Input publication. No full repository suite was run.

The parent found and corrected three safety details: a human choice must not
bridge a gap at breast height; a returned task must match the submitted request
ID; and response-body reads must remain within the request deadline. Rendering
failure prevents confirmation. Those conditions are covered by focused tests.

Actual source desktop workflow hydration and HTTP preparation exposed the correct
five unavailable IDs. A read-only browser check showed T-1-006's current crown,
80,000 raw points and all three axes. At 150% display scaling, the first page
revealed a canvas/SVG size mismatch. Scoped canvas CSS corrected both to
880 × 608.67 CSS pixels; the corrected screenshot shows axes aligned with raw
stems. No candidate or seed was selected, and the confirmation checkbox remained
unchecked with the submit button disabled. The finite audit server stopped after
600 seconds. Final request-protocol refinements were validated by the focused
UI tests; this is not a packaged/native-window or independent-PC acceptance.

The existing viewer diagnostic endpoint updated `logs/viewer_init.latest.json`
during visual checking. This did not touch science. The newly generated diagnostic
is preserved in V4 evidence; the original bytes were restored from the original
project ZIP, checked against its member CRC and the last preexisting application-log
event, with checkpoint mtime restored. `viewer_diagnostic_restoration.json` records
the exact source and both hashes. The application log itself was never changed.
The audit helper now redirects this endpoint too.

Review decisions remain separate from memberships and are bound to source,
editor revision and current DBH generation. Accepted review proof survives read-only
restart; later editing invalidates it. An unchanged V3 finalize retains reviewed
stem reservations and their proof dependencies. No fit threshold, Ground/HAG/Wire,
crown segmentation, height, clearance or risk formula was edited.

## Human acceptance remains pending

Use the source application and instructions in `DBH_RECOVERY_V4_REVIEW_GUIDE.zh-CN.md`.
The operator must inspect the candidate/seed views and confirm only genuine stem
identity. The system then calculates DBH and may still reject poor geometry.
After actual decisions, rerun DBH/publication acceptance; do not infer 12/12 from
implementation tests. No package rebuild and no push were performed.
